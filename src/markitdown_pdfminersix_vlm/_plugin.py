import io
import logging
import copy
from typing import BinaryIO, Any
import config

import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError

from markitdown import (
    MarkItDown,
    DocumentConverter,
    DocumentConverterResult,
    StreamInfo
)

from .utils.high_level import (
    get_pdf_pages_count,
    extract_text
)

__plugin_inferface_version__ = (
    1
)

ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/pdf",
]

ACCEPTED_FILE_EXTENSIONS = ["*.pdf"]

def register_converters(markitdown: MarkItDown, **kwargs) -> None:
    markitdown.register_converter(EPdfConverter())

class EPdfConverter(DocumentConverter):
    """
    Convert a document to PDF format.
    """

    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any
    ) -> bool:
        """
        Check if the converter can handle the given file stream.
        """
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()
        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any
    ) -> DocumentConverterResult:
        """
        Convert the pdf document to markdown.
        """
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_THREADS)

            num_pages = get_pdf_pages_count(file_stream)
            self.logger.info(f"** Generating supplementary content from {stream_info.filename}")

            # Map futures to their corresponding page ranges
            file_content = file_stream.read()
            future_to_page_range = {}
            for page_number in range(0, num_pages, config.MAX_PAGES_PER_THREAD):
                future_file_stream = io.BytesIO(file_content)
                future = executor.submit(
                    extract_text,
                    future_file_stream,
                    page_numbers=range(page_number, min(page_number + config.MAX_PAGES_PER_THREAD, num_pages)),
                )
                future_to_page_range[future] = page_number

            # Initialize a list to store results in order
            results = [None] * len(future_to_page_range)

            # Process futures as they complete
            for future in concurrent.futures.as_completed(future_to_page_range):
                page_range_start = future_to_page_range[future]
                try:
                    page_text = future.result(timeout=150)
                    if page_text:
                        # Store the result in the correct position
                        results[page_range_start // config.MAX_PAGES_PER_THREAD] = page_text
                except FuturesTimeoutError:
                    self.logger.error(
                        f"** Error while processing page range starting at {page_range_start} concurrently",
                        exc_info=config.FULL_LOGGING,
                    )
                    continue
            
            # Combine results in order
            output = "\n".join(results)
            return DocumentConverterResult(
                title=None,
                markdown=output)
        except Exception as err:
            self.logger.error(
                f"** Error while processing {stream_info.filename} due to {err}", 
                exc_info=config.FULL_LOGGING)