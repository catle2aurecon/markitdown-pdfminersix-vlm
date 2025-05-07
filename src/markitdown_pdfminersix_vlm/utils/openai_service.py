# app/services/openai_service.py
import json
from typing import Any, Dict
import logging

from openai import AzureOpenAI

import config

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        )
        self.azure_deployment = config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.logger = logging.getLogger(__name__)

    def generate(
        self,
        prompt,
        system_message: str = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
        response_format="text",
        output_schema=None,
    ) -> Dict[str, Any]:

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})
            # Configure response format
            if response_format == "json_object":
                response_format_param = {"type": "json_object"}
                if output_schema:
                    # OpenAI uses functions for schema validation
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "generate_response",
                                "description": "Generate a response to the user query",
                                "parameters": output_schema,
                            },
                        }
                    ]

                    # Set tool_choice to use our function
                    tool_choice = {
                        "type": "function",
                        "function": {"name": "generate_response"},
                    }

                    response = self.client.chat.completions.create(
                        model=self.azure_deployment,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    # Extract response from tool_calls
                    if response.choices and response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]

                        return {"success": True, "response": json.loads(tool_call.function.arguments)}
                    else:
                        return {"success": False, "response": "No tool calls found in response"}
                else:
                    # Use regular response_format for JSON
                    response = self.client.chat.completions.create(
                        model=self.azure_deployment,
                        messages=messages,
                        response_format=response_format_param,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    return {"success": True, "response": json.loads(response.choices[0].message.content)}
            else:
                # Regular text response
                response = self.client.chat.completions.create(
                    model=self.azure_deployment, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                completion = response.choices[0].message.content.strip()
                return {"success": True, "response": completion}

        except Exception as err:
            self.logger.error(f"Error in Azure OpenAI API call: {str(err)}", exc_info=config.FULL_LOGGING)
            raise Exception(message="Failed to generate completion", status_code=500) from err
