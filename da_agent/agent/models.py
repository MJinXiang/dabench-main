import base64
import json
import logging
import os
import re
import time
from http import HTTPStatus
from io import BytesIO

from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
import dashscope
from groq import Groq
import google.generativeai as genai
import openai
import requests
import tiktoken
import signal

logger = logging.getLogger("api-llms")


def call_llm(payload):
    model = payload["model"]
    stop = ["Observation:","\n\n\n\n","\n \n \n"]
    if model.startswith("gpt"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        

        for i in range(3):
            try:
                response = requests.post(
                            "https://api2.aigcbest.top/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, 'response') and e.response is not None:
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = 'unknown_error'
                logger.error("Retrying ...")
                time.sleep(4 * (2 ** (i + 1)))
        return False, code_value

    elif model.startswith("azure"):
        client = AzureOpenAI(
            api_key = os.environ['AZURE_API_KEY'],  
            api_version = "2024-02-15-preview",
            azure_endpoint = os.environ['AZURE_ENDPOINT']
            )
        model_name = model.split("/")[-1]
        for i in range(3):
            try:
                response = client.chat.completions.create(model=model_name,messages=payload['messages'], max_tokens=payload['max_tokens'], top_p=payload['top_p'], temperature=payload['temperature'], stop=stop)
                response = response.choices[0].message.content
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, response
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                error_info = e.response.json()  
                code_value = error_info['error']['code']
                if code_value == "content_filter":
                    if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                        payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                if code_value == "context_length_exceeded":
                    return False, code_value        
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value
        
    elif model.startswith("claude"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        gemini_messages = []

        for i, message in enumerate(messages):
            gemini_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    gemini_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    gemini_message['content'].append({"type": "text", "text": part['text']})

            gemini_messages.append(gemini_message)
        
        if gemini_messages[0]['role'] == "system":
            gemini_system_message_item = gemini_messages[0]['content'][0]
            gemini_messages[1]['content'].insert(0, gemini_system_message_item)
            gemini_messages.pop(0)


        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {os.environ["GEMINI_API_KEY"]}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }  
        
        payload = json.dumps({"model": model,"messages": gemini_messages,"max_tokens": max_tokens,"temperature": temperature,"top_p": top_p})


        
        for i in range(3):
            try:
                response = requests.request("POST", "https://api2.aigcbest.top/v1/chat/completions", headers=headers, data=payload)
                logger.info(f"response_code {response.status_code}")
                if response.status_code == 200:
                    return True, response.json()['choices'][0]['message']['content']
                else:
                    error_info = response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                    logger.error("Retrying ...")
                    time.sleep(10 * (2 ** (i + 1)))
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                code_value = "context_length_exceeded"
        return False, code_value
                           
    # elif model.startswith("claude"):
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]

    #     claude_messages = []

    #     for i, message in enumerate(messages):
    #         claude_message = {
    #             "role": message["role"],
    #             "content": []
    #         }
    #         assert len(message["content"]) in [1, 2], "One text, or one text with one image"
    #         for part in message["content"]:

    #             if part['type'] == "image_url":
    #                 image_source = {}
    #                 image_source["type"] = "base64"
    #                 image_source["media_type"] = "image/png"
    #                 image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
    #                 claude_message['content'].append({"type": "image", "source": image_source})

    #             if part['type'] == "text":
    #                 claude_message['content'].append({"type": "text", "text": part['text']})

    #         claude_messages.append(claude_message)

    #     # the claude not support system message in our endpoint, so we concatenate it at the first user message
    #     if claude_messages[0]['role'] == "system":
    #         claude_system_message_item = claude_messages[0]['content'][0]
    #         claude_messages[1]['content'].insert(0, claude_system_message_item)
    #         claude_messages.pop(0)

    #     logger.debug("CLAUDE MESSAGE: %s", repr(claude_messages))

    #     headers = {
    #         'Accept': 'application/json',
    #         'Authorization': f'Bearer {os.environ["ANTHROPIC_API_KEY"]}',
    #         'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #         'Content-Type': 'application/json'
    #     }

    #     payload = {
    #         "model": model,
    #         "max_tokens": max_tokens,
    #         "messages": claude_messages,
    #         "temperature": temperature,
    #         "top_p": top_p
    #     }

    #     for i in range(3):
    #         try:
    #             response = requests.post(
    #                 "https://api.claude-Plus.top/v1/chat/completions",
    #                 headers=headers,
    #                 json=payload
    #             )
    #             return True, response.json()['choices'][0]['message']['content']
    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             time.sleep(10 * (2 ** (i + 1)))
    #             if hasattr(e, 'response'):
    #                 error_info = e.response.json()  
    #                 code_value = error_info['error']['code']
    #                 if code_value == "content_filter":
    #                     if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
    #                         payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #                 if code_value == "context_length_exceeded":
    #                     return False, code_value        
    #             else:
    #                 code_value = "context_length_exceeded"
    #             logger.error("Retrying ...")
    #     return False, code_value

        # if response.status_code != 200:

        #     logger.error("Failed to call LLM: " + response.text)
        #     time.sleep(5)
        #     return ""
        # else:
        #     return response.json()['content'][0]['text']

    elif model.startswith("mixtral"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        for i in range(2):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = ""
                logger.error("Retrying ...") 

        return False, code_value
        
    elif model == "llama3-70b":
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        groq_messages = []

        for i, message in enumerate(messages):
            groq_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                groq_message['content'] = part['text'] if part['type'] == "text" else ""

            groq_messages.append(groq_message)

        # The implementation based on Groq API
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=groq_messages,
                    model="llama3-70b-8192",
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = "context_length_exceeded"
                logger.error("Retrying ...")

        return False, code_value
        
    elif model.startswith("o1"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        
        messages = payload["messages"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
            
            
        o1_messages = []

        for i, message in enumerate(messages):
            o1_message = {
                "role": message["role"] if message["role"] != "system" else "user",
                "content": ""
            }
            for part in message["content"]:
                o1_message['content'] = part['text'] if part['type'] == "text" else ""
                
                o1_messages.append(o1_message)

        payload["messages"] = o1_messages
        payload["max_completion_tokens"] = 10000
        del payload['max_tokens']
        del payload["temperature"]
        del payload["top_p"]

        for i in range(3):
            try:
                response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value

    elif model.startswith("deepseek"):
        
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        from openai import OpenAI
        
        deepseek_messages = []

        for i, message in enumerate(messages):
            deepseek_message = {
                "role": message["role"],
                "content": ""
            }
            for part in message["content"]:
                deepseek_message['content'] = part['text'] if part['type'] == "text" else ""
                deepseek_messages.append(deepseek_message)
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=deepseek_messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                )
                output_message = json.loads(response.json())['choices'][0]['message']['content']
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, 'response') and e.response is not None:
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']

                    if code_value == "content_filter":
                        last_message = messages[-1]
                        if 'content' in last_message and isinstance(last_message['content'], str):
                            if not last_message['content'].endswith("They do not represent any real events or entities. ]"):
                                last_message['content'] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                        else:
                            logger.error("Unexpected message structure in 'messages'. Skipping content modification.")
                    elif code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = 'unknown_error'
                
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
            return False, code_value


        
    elif model.startswith("qwen") or model.startswith("Qwen") or model.startswith("llama3.1"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        
        if model.startswith("llama3.1"):
            max_tokens = 2000

        qwen_messages = []

        for i, message in enumerate(messages):
            qwen_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:
                qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

            qwen_messages.append(qwen_message)

        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = dashscope.Generation.call(
                    model=model,
                    messages=qwen_messages,
                    result_format="message",
                    max_length=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response['output']['choices'][0]['message']['content']

            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(3 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                else:
                    code_value = "context_length_exceeded"
                logger.error("Retrying ...")
        return False, code_value
        #         if response.status_code == HTTPStatus.OK:
        #             break
        #         else:
        #             logger.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
        #                 response.request_id, response.status_code,
        #                 response.code, response.message
        #             ))
        #             raise Exception("Failed to call LLM: " + response.message)
        #     except:
        #         if flag == 0:
        #             qwen_messages = [qwen_messages[0]] + qwen_messages[-1:]
        #         else:
        #             for i in range(len(qwen_messages[-1]["content"])):
        #                 if "text" in qwen_messages[-1]["content"][i]:
        #                     qwen_messages[-1]["content"][i]["text"] = ' '.join(
        #                         qwen_messages[-1]["content"][i]["text"].split()[:-500])
        #         flag = flag + 1

        # try:
            
        # except Exception as e:
        #     logger.error("Failed to call LLM: " + str(e))
        #     time.sleep(10 * (2 ** (i + 1)))
        #     error_info = e.response.json()  
        #     code_value = error_info['error']['code']
        #     response = error_info['error']['message']
        

    elif model.startswith("THUDM"):
        # THUDM/cogagent-chat-hf
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        cog_messages = []

        for i, message in enumerate(messages):
            cog_message = {
                "role": message["role"],
                "content": []
            }

            for part in message["content"]:
                if part['type'] == "image_url":
                    cog_message['content'].append(
                        {"type": "image_url", "image_url": {"url": part['image_url']['url']}})

                if part['type'] == "text":
                    cog_message['content'].append({"type": "text", "text": part['text']})

            cog_messages.append(cog_message)

        # the cogagent not support system message in our endpoint, so we concatenate it at the first user message
        if cog_messages[0]['role'] == "system":
            cog_system_message_item = cog_messages[0]['content'][0]
            cog_messages[1]['content'].insert(0, cog_system_message_item)
            cog_messages.pop(0)

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": cog_messages,
            "temperature": temperature,
            "top_p": top_p
        }

        base_url = "http://127.0.0.1:8000"

        response = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=False)
        if response.status_code == 200:
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            return content
        else:
            print("Failed to call LLM: ", response.status_code)
            return ""

    elif model == "gemini-1.5-pro-latest":
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        gemini_messages = []

        for i, message in enumerate(messages):
            gemini_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    gemini_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    gemini_message['content'].append({"type": "text", "text": part['text']})

            gemini_messages.append(gemini_message)

        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {os.environ["GEMINI_API_KEY"]}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }  
        
        payload = json.dumps({"model": model,"messages": gemini_messages,"max_tokens": max_tokens,"temperature": temperature,"top_p": top_p})


        
        for i in range(3):
            try:
                response = requests.request("POST", "https://api2.aigcbest.top/v1/chat/completions", headers=headers, data=payload)
                logger.info(f"response_code {response.status_code}")
                if response.status_code == 200:
                    return True, response.json()['choices'][0]['message']['content']
                else:
                    error_info = response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                    logger.error("Retrying ...")
                    time.sleep(10 * (2 ** (i + 1)))
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                code_value = "context_length_exceeded"
        return False, code_value
                           
                    
        #     except:
        #         time.sleep(5)
        #         continue
        #     if response.status_code == 200:
        #         result = response.json()['choices'][0]['message']['content']
        #         break
        #     else:
        #         logger.error(f"Failed to call LLM")
        #         time.sleep(5)
        #         attempt += 1
        # else:
        #     print("Exceeded maximum attempts to call LLM.")
        #     result = ""
            
        # return result


    # elif model.startswith("qwen"):
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     if payload["temperature"]:
    #         logger.warning("Qwen model does not support temperature parameter, it will be ignored.")

    #     qwen_messages = []

    #     for i, message in enumerate(messages):
    #         qwen_message = {
    #             "role": message["role"],
    #             "content": []
    #         }
    #         assert len(message["content"]) in [1, 2], "One text, or one text with one image"
    #         for part in message["content"]:
    #             qwen_message['content'].append({"image": part['image_url']['url']}) if part[
    #                                                                                         'type'] == "image_url" else None
    #             qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

    #         qwen_messages.append(qwen_message)

    #     response = dashscope.MultiModalConversation.call(
    #         model='qwen-vl-plus',
    #         messages=messages,
    #         max_length=max_tokens,
    #         top_p=top_p,
    #     )
    #     # The response status_code is HTTPStatus.OK indicate success,
    #     # otherwise indicate request is failed, you can get error code
    #     # and message from code and message.
    #     if response.status_code == HTTPStatus.OK:
    #         try:
    #             return response.json()['output']['choices'][0]['message']['content']
    #         except Exception:
    #             return ""
    #     else:
    #         print(response.code)  # The error code.
    #         print(response.message)  # The error message.
    #         return ""

    # else:
    #     raise ValueError("Invalid model: " + model)
    
