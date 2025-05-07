import io
import logging
import re
from typing import (
    BinaryIO,
    Generic,
    List,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    cast,
)

from pdfminer import utils
from pdfminer.image import ImageWriter
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTContainer,
    LTCurve,
    LTFigure,
    LTImage,
    LTItem,
    LTLayoutContainer,
    LTLine,
    LTPage,
    LTRect,
    LTText,
    LTTextBox,
)
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfdevice import PDFTextDevice
from pdfminer.pdffont import PDFFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import PDFStream
from pdfminer.utils import (
    AnyIO,
    Matrix,
    PathSegment,
    Point,
    Rect,
    apply_matrix_pt,
    mult_matrix,
)

from utils.image import ImageSummarizer

log = logging.getLogger(__name__)


class PDFLayoutAnalyzer(PDFTextDevice):
    cur_item: LTLayoutContainer
    ctm: Matrix

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
    ) -> None:
        PDFTextDevice.__init__(self, rsrcmgr)
        self.pageno = pageno
        self.laparams = laparams
        self._stack: List[LTLayoutContainer] = []

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        (x0, y0, x1, y1) = page.mediabox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(self.pageno, mediabox)

    def end_page(self, page: PDFPage) -> None:
        assert not self._stack, str(len(self._stack))
        assert isinstance(self.cur_item, LTPage), str(type(self.cur_item))
        if self.laparams is not None:
            self.cur_item.analyze(self.laparams)
        self.pageno += 1
        self.receive_layout(self.cur_item)

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))

    def end_figure(self, _: str) -> None:
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)

    def render_image(self, name: str, stream: PDFStream) -> None:
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        item = LTImage(
            name,
            stream,
            (self.cur_item.x0, self.cur_item.y0, self.cur_item.x1, self.cur_item.y1),
        )
        self.cur_item.add(item)

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
    ) -> None:
        """Paint paths described in section 4.4 of the PDF reference manual"""
        shape = "".join(x[0] for x in path)

        if shape[:1] != "m":
            # Per PDF Reference Section 4.4.1, "path construction operators may
            # be invoked in any sequence, but the first one invoked must be m
            # or re to begin a new subpath." Since pdfminer.six already
            # converts all `re` (rectangle) operators to their equivelent
            # `mlllh` representation, paths ingested by `.paint_path(...)` that
            # do not begin with the `m` operator are invalid.
            pass

        elif shape.count("m") > 1:
            # recurse if there are multiple m's in this shape
            for m in re.finditer(r"m[^m]+", shape):
                subpath = path[m.start(0):m.end(0)]
                self.paint_path(gstate, stroke, fill, evenodd, subpath)

        else:
            # Although the 'h' command does not not literally provide a
            # point-position, its position is (by definition) equal to the
            # subpath's starting point.
            #
            # And, per Section 4.4's Table 4.9, all other path commands place
            # their point-position in their final two arguments. (Any preceding
            # arguments represent control points on Bézier curves.)
            raw_pts = [cast(Point, p[-2:] if p[0] != "h" else path[0][-2:]) for p in path]
            pts = [apply_matrix_pt(self.ctm, pt) for pt in raw_pts]

            operators = [str(operation[0]) for operation in path]
            transformed_points = [
                [
                    apply_matrix_pt(self.ctm, (float(operand1), float(operand2)))
                    for operand1, operand2 in zip(operation[1::2], operation[2::2])
                ]
                for operation in path
            ]
            transformed_path = [cast(PathSegment, (o, *p)) for o, p in zip(operators, transformed_points)]

            # Drop a redundant "l" on a path closed with "h"
            if len(shape) > 3 and shape[-2:] == "lh" and pts[-2] == pts[0]:
                shape = shape[:-2] + "h"
                pts.pop()

            if shape in {"mlh", "ml"}:
                # single line segment
                #
                # Note: 'ml', in conditional above, is a frequent anomaly
                # that we want to support.
                line = LTLine(
                    gstate.linewidth,
                    pts[0],
                    pts[1],
                    stroke,
                    fill,
                    evenodd,
                    gstate.scolor,
                    gstate.ncolor,
                    original_path=transformed_path,
                    dashing_style=gstate.dash,
                )
                self.cur_item.add(line)

            elif shape in {"mlllh", "mllll"}:
                (x0, y0), (x1, y1), (x2, y2), (x3, y3), _ = pts

                is_closed_loop = pts[0] == pts[4]
                has_square_coordinates = (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0) or (
                    y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0
                )
                if is_closed_loop and has_square_coordinates:
                    rect = LTRect(
                        gstate.linewidth,
                        (*pts[0], *pts[2]),
                        stroke,
                        fill,
                        evenodd,
                        gstate.scolor,
                        gstate.ncolor,
                        transformed_path,
                        gstate.dash,
                    )
                    self.cur_item.add(rect)
                else:
                    curve = LTCurve(
                        gstate.linewidth,
                        pts,
                        stroke,
                        fill,
                        evenodd,
                        gstate.scolor,
                        gstate.ncolor,
                        transformed_path,
                        gstate.dash,
                    )
                    self.cur_item.add(curve)
            else:
                curve = LTCurve(
                    gstate.linewidth,
                    pts,
                    stroke,
                    fill,
                    evenodd,
                    gstate.scolor,
                    gstate.ncolor,
                    transformed_path,
                    gstate.dash,
                )
                self.cur_item.add(curve)

    def render_char(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
    ) -> float:
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        return item.adv

    def handle_undefined_char(self, font: PDFFont, cid: int) -> str:
        log.debug("undefined: %r, %r", font, cid)
        return "(cid:%d)" % cid

    def receive_layout(self, ltpage: LTPage) -> None:
        pass


class PDFPageAggregator(PDFLayoutAnalyzer):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.result: Optional[LTPage] = None

    def receive_layout(self, ltpage: LTPage) -> None:
        self.result = ltpage

    def get_result(self) -> LTPage:
        assert self.result is not None
        return self.result


# Some PDFConverter children support only binary I/O
IOType = TypeVar("IOType", TextIO, BinaryIO, AnyIO)


class PDFConverter(PDFLayoutAnalyzer, Generic[IOType]):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: IOType,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.outfp: IOType = outfp
        self.codec = codec
        self.outfp_binary = self._is_binary_stream(self.outfp)

    @staticmethod
    def _is_binary_stream(outfp: AnyIO) -> bool:
        """Test if an stream is binary or not"""
        if "b" in getattr(outfp, "mode", ""):
            return True
        elif hasattr(outfp, "mode"):
            # output stream has a mode, but it does not contain 'b'
            return False
        elif isinstance(outfp, io.BytesIO):
            return True
        elif isinstance(outfp, io.StringIO) or isinstance(outfp, io.TextIOBase):
            return False

        return True


class TextConverter(PDFConverter[AnyIO]):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
        showpageno: bool = False,
        imagewriter: Optional[ImageWriter] = None,
        imagesummarizer: Optional[ImageSummarizer] = None,
    ) -> None:
        super().__init__(rsrcmgr, outfp, codec=codec, pageno=pageno, laparams=laparams)
        self.showpageno = showpageno
        self.imagewriter = imagewriter
        self.imagesummarizer = imagesummarizer

    def write_text(self, text: str) -> None:
        text = utils.compatible_encode_method(text, self.codec, "ignore")
        if self.outfp_binary:
            cast(BinaryIO, self.outfp).write(text.encode())
        else:
            cast(TextIO, self.outfp).write(text)

    def receive_layout(self, ltpage: LTPage) -> None:
        def render(item: LTItem) -> None:
            if isinstance(item, LTContainer):
                for child in item:
                    render(child)
            elif isinstance(item, LTText):
                self.write_text(item.get_text())
            if isinstance(item, LTTextBox):
                self.write_text("\n")
            elif isinstance(item, LTImage):
                item_name = None
                if self.imagewriter is not None:
                    item_name = self.imagewriter.export_image(item)
                if self.imagesummarizer is not None:
                    self.write_text(
                        "\n"
                        + self.imagesummarizer.summarize_file(f"{self.imagewriter.outdir}/{item_name}")
                        # self.imagesummarizer.summarize_fp(item)
                    )

        if self.showpageno:
            self.write_text(f"Page {ltpage.pageid}\n")
        render(ltpage)
        self.write_text("\f")

    # Some dummy functions to save memory/CPU when all that is wanted
    # is text.  This stops all the image and drawing output from being
    # recorded and taking up RAM.
    def render_image(self, name: str, stream: PDFStream) -> None:
        if self.imagewriter is not None:
            PDFConverter.render_image(self, name, stream)

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
    ) -> None:
        pass
