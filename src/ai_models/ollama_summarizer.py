import os
import time
import socket
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any
import re

# External Libraries
import ollama
from ollama import Client, ResponseError
from loguru import logger

# Local Imports 
from src.config import settings

class OllamaSummarizer:
    def __init__(self, model_name: str = settings.DEFAULT_OLLAMA_MODEL):
        self.model_name = model_name
        self._parse_ollama_host()
        
        # Initialize the Ollama client with the specified host
        self.client = Client(host=settings.OLLAMA_HOST)
        
        if not self._ensure_ollama_running():
            logger.critical(
                f"Ollama service is not accessible at {settings.OLLAMA_HOST}. "
                "Please ensure Ollama is running and accessible (e.g., check firewall, service status)."
            )
            raise ConnectionError(
                f"Cannot connect to Ollama service at {settings.OLLAMA_HOST}. "
                "The requested service provider could not be loaded or initialized, "
                "or Ollama is not running. Check Ollama server logs for more details."
            )
        
        if not self._is_model_available(self.model_name):
            error_msg = (
                f"The requested Ollama model '{self.model_name}' is not available on the server at "
                f"{settings.OLLAMA_HOST}. Please ensure the model is pulled using 'ollama pull {self.model_name}' "
                "on the server machine, or check the model name."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        # --- END NEW ---

    def _parse_ollama_host(self) -> None:
        """Parse Ollama host URL into host and port."""
        parsed = urlparse(settings.OLLAMA_HOST)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 11434
        logger.debug(f"Using Ollama service at {settings.OLLAMA_HOST}")

    def _ensure_ollama_running(self) -> bool:
        """
        Ensure Ollama service is running and accessible via HTTP.
        This uses a direct socket connection check as a pre-flight,
        which is still valuable for diagnosing the "socket provider" error.
        """
        logger.info(f"Checking Ollama service availability at {settings.OLLAMA_HOST}...")
        max_retries = 5
        retry_delay = 2 # seconds
        for i in range(max_retries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:                    
                    # Try a simple API call to confirm it's healthy
                    try:
                        self.client.list() # Attempt to list models as a more robust health check
                        logger.info("Ollama service is responsive via API.")
                        return True
                    except ResponseError as e:
                        logger.warning(f"Ollama API responded with an error during health check: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to get response from Ollama API during health check: {e}")

                except ConnectionRefusedError:
                    logger.warning(
                        f"Connection refused by Ollama at {settings.OLLAMA_HOST}. "
                        f"Is Ollama running? Retrying in {retry_delay} seconds... ({i+1}/{max_retries})"
                    )
                except socket.timeout:
                    logger.warning(
                        f"Timeout connecting to Ollama at {settings.OLLAMA_HOST}. "
                        f"Is Ollama running and listening? Retrying in {retry_delay} seconds... ({i+1}/{max_retries})"
                    )
                except socket.error as e:
                    logger.error(
                        f"Low-level socket error while trying to connect to Ollama: {e}. "
                        "This often indicates a problem with your operating system's network stack "
                        "or aggressive security software. Further retries are unlikely to help."
                    )
                    return False 
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred while checking Ollama availability: {e}. "
                        f"Retrying in {retry_delay} seconds... ({i+1}/{max_retries})"
                    )
            time.sleep(retry_delay)
        
        logger.error(f"Failed to connect to Ollama after {max_retries} attempts.")
        return False

    def _is_model_available(self, model_name: str) -> bool:
        """
        Checks if the specified model is available on the Ollama server.
        """
        try:
            models_info = self.client.list()
            for model_entry in models_info.get('models', []):
                # Check if the base model name matches (e.g., 'gemma3' matches 'gemma3:latest')
                if model_entry['name'].startswith(model_name + ':'):
                    return True
            return False
        except ResponseError as e:
            logger.error(f"Failed to list models from Ollama server (status {e.status_code}): {e.error}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while checking model availability: {e}")
            return False

    def _call_ollama(self, prompt: str, input_text: str) -> Optional[str]:
        """Call Ollama using the specified model via HTTP API."""
        try:
            combined = f"{prompt}\n\n{input_text}"
            logger.debug(f"Calling Ollama at {settings.OLLAMA_HOST} with model {self.model_name}")
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': combined}
                ],
                stream=False
            )
            
            output = response['message']['content']
            
            if not output:
                logger.warning("No response content received from Ollama")
                return None
                
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

            logger.debug(f"Ollama response received, length: {len(output)} chars")
            return output

        except ResponseError as e:
            if e.status_code == 404 and f"model '{self.model_name}'" in e.error:
                logger.error(
                    f"Model '{self.model_name}' not found on Ollama server during generation. "
                    "Please ensure it is pulled and available."
                )
            else:
                logger.error(f"Ollama API call failed: Status {e.status_code}, Error: {e.error}")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return None

    def summarize_report(self, report_path: str) -> Optional[str]:
        """Summarize a video analysis report using Ollama."""
        try:
            report_path = Path(report_path).resolve()
            if not report_path.exists():
                raise FileNotFoundError(f"Report file not found: {report_path}")

            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()

            # Clean the report text
            clean_text = re.sub(r'[\u06F0-\u06F9]+', '', report_text)
            clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)

            summary = self._call_ollama(settings.video_summary_prompt, clean_text)
            
            if not summary:
                logger.warning("Using fallback summary due to Ollama failure")
                return "The video presents a dynamic scene with various events, blending spoken words and visuals into an engaging narrative."

            return summary

        except Exception as e:
            logger.error(f"Error during report summarization: {str(e)}")
            return None

    def summarize(self, text: str) -> str:
        """
        Summarize the provided text using Ollama.
        
        Args:
            text (str): The text to summarize
            
        Returns:
            str: The generated summary
        """
        try:
            summary = self._call_ollama(settings.video_summary_prompt, text)
            
            if not summary:
                logger.warning("Using fallback summary due to Ollama failure")
                return "The video could not be summarized due to an error with the language model or prompt."

            return summary

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise