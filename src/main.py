import os
from _plugin import EPdfConverter
from markitdown import StreamInfo

def test_pdf_converter():
    test_file = os.path.join(os.path.dirname(__file__),'./tests/data/test.pdf')
    with open(test_file, 'rb') as file_stream:
        converter = EPdfConverter()
        converter.convert(
            file_stream=file_stream,
            stream_info=StreamInfo(
                mimetype="application/pdf",
                filename="test.pdf",
                extension=".pdf"
            )
        )
        
if __name__ == "__main__":
    test_pdf_converter()