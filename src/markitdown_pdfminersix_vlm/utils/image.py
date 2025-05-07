import base64
from io import BytesIO
from PIL import Image
from .openai_service import AzureOpenAIService


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type = 'image/jpeg'  # Default MIME type if none is found

    # Read and encode the image file
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_encoded_data = base64.b64encode(buffer.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


class ImageSummarizer:
    def __init__(self, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.azure_openai_service = AzureOpenAIService()

    def summarize_file(self, item_path: str) -> str:
        # Placeholder for image summarization logic
        data_url = local_image_to_data_url(item_path)
        prompt = [
            {"type": "text", "text": self.user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            },
        ]
        output = self.azure_openai_service.generate(
            prompt,
            system_message=self.system_prompt,
            max_tokens=2000
        )

        return output["response"]
