# pylint: disable=W0613
"""Functions that can be used for the most common use-cases for pdfminer.six"""

import logging
import sys
from io import StringIO
from typing import Any, BinaryIO, Container, Optional, cast

from pdfminer.image import ImageWriter
from pdfminer.layout import LAParams
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import AnyIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import resolve1

from utils.converter import TextConverter
from utils.image import ImageSummarizer

OUTPUT_TYPES = ((".htm", "html"), (".html", "html"), (".xml", "xml"), (".tag", "tag"))


def extract_text_to_fp(
    inf: BinaryIO,
    outfp: AnyIO,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: Optional[LAParams] = None,
    maxpages: int = 0,
    page_numbers: Optional[Container[int]] = None,
    password: str = "",
    scale: float = 1.0,
    rotation: int = 0,
    layoutmode: str = "normal",
    output_dir: Optional[str] = "/tmp",
    strip_control: bool = False,
    debug: bool = False,
    disable_caching: bool = False,
    **kwargs: Any,
) -> None:
    """Parses text from inf-file and writes to outfp file-like object.

    Takes loads of optional arguments but the defaults are somewhat sane.
    Beware laparams: Including an empty LAParams is not the same as passing
    None!

    :param inf: a file-like object to read PDF structure from, such as a
        file handler (using the builtin `open()` function) or a `BytesIO`.
    :param outfp: a file-like object to write the text to.
    :param output_type: May be 'text', 'xml', 'html', 'hocr', 'tag'.
        Only 'text' works properly.
    :param codec: Text decoding codec
    :param laparams: An LAParams object from pdfminer.layout. Default is None
        but may not layout correctly.
    :param maxpages: How many pages to stop parsing after
    :param page_numbers: zero-indexed page numbers to operate on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param scale: Scale factor
    :param rotation: Rotation factor
    :param layoutmode: Default is 'normal', see
        pdfminer.converter.HTMLConverter
    :param output_dir: If given, creates an ImageWriter for extracted images.
    :param strip_control: Does what it says on the tin
    :param debug: Output more logging data
    :param disable_caching: Does what it says on the tin
    :param other:
    :return: nothing, acting as it does on two streams. Use StringIO to get
        strings.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)

    imagesummarizer = ImageSummarizer(
        system_prompt="You are a helpful assistant that summarizes images.",
        user_prompt="Summarize the image.",
    )

    rsrcmgr = PDFResourceManager(caching=not disable_caching)
    device: Optional[PDFDevice] = None

    if output_type != "text" and outfp == sys.stdout:
        outfp = sys.stdout.buffer

    if output_type == "text":
        device = TextConverter(
            rsrcmgr, outfp, codec=codec, laparams=laparams, imagewriter=imagewriter, imagesummarizer=imagesummarizer
        )

    elif output_type == "tag":
        # Binary I/O is required, but we have no good way to test it here.
        device = TagExtractor(rsrcmgr, cast(BinaryIO, outfp), codec=codec)

    else:
        msg = f"Output type can be text, html, xml or tag but is {output_type}"
        raise PDFValueError(msg)

    assert device is not None
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(
        inf,
        page_numbers,
        maxpages=maxpages,
        password=password,
        caching=not disable_caching,
    ):
        page.rotate = (page.rotate + rotation) % 360
        interpreter.process_page(page)

    device.close()


def extract_text(
    fp: BinaryIO,
    laparams: Optional[LAParams] = None,
    codec: str = "utf-8",
    strip_control: bool = False,
    maxpages: int = 0,
    page_numbers: Optional[Container[int]] = None,
    password: str = "",
    scale: float = 1.0,
    rotation: int = 0,
    layoutmode: str = "normal",
    output_dir: Optional[str] = "/tmp",
    debug: bool = False,
    disable_caching: bool = False,
    **kwargs: Any,
) -> AnyIO:
    outfp = StringIO()
    extract_text_to_fp(fp, **locals())
    return outfp.getvalue()


def get_pdf_pages_count(file_stream) -> int:
    """
    Get the number of pages in a PDF file using pdfminer.

    Args:
        file_stream: A file-like object containing the PDF data

    Returns:
        int: Number of pages in the PDF
    """
    try:
        # Create a PDF parser object
        parser = PDFParser(file_stream)

        # Create a PDF document object
        document = PDFDocument(parser)

        # Get the number of pages
        num_pages = resolve1(document.catalog["Pages"])["Count"]

        # Reset the file stream position
        file_stream.seek(0)

        return num_pages

    except Exception as e:
        print(f"** Error reading PDF pages: {str(e)}")
        return 0
