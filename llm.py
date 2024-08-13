import re
import os
import time
import json
import demjson3
import ollama

# Set the environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# OLLAMA host & model
OLLAMA_HOST  = 'http://192.168.50.81:11434' #'http://localhost:11434'
OLLAMA_MODEL =  'gemma2:27b' #'lucas2024/gemma-2-9b-it-sppo-iter3:q8_0' #'llama3.1' #'llama3.1:70b' 
OLLAMA_CONTEXT_SIZE = 8_000 # Max context size for OLLAMA is 

def initialize_history():
    # Load the system_prompt from file
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, 'prompt_system.txt'), 'r') as file:
        system_prompt = file.read()

    history = [{'role': 'system', 'content': f'{system_prompt}'}]
    
    return history

def detect_fallacies(text_path, fallacy_analysis_path, temp_file):
    history = initialize_history()
    client = ollama.Client(host=OLLAMA_HOST)
    
    llm_outputs = []
    
    # Load from temp file if it exists
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            history = temp_data.get('history', history)
            llm_outputs = temp_data.get('llm_outputs', [])
            processed_lines = temp_data.get('processed_lines', [])
            print("Loaded from history from temp file:", temp_file)
    else:
        processed_lines = []

    for line in read_line_from_file(text_path):
        if line in processed_lines:
            continue
        history, llm_outputs = prompt_ollama(line, history, llm_outputs, client)
        processed_lines.append(line)
        save_intermediate_results(temp_file, history, llm_outputs, processed_lines)
    
    save_llm_output_to_json(llm_outputs, fallacy_analysis_path)
    # Remove the temp file after completion
    if os.path.exists(temp_file):
        os.remove(temp_file)

def prompt_ollama(line, history, llm_outputs, client, pre_prompt='Input Text:'):
    max_retries = 3
    retry_count = 0
    valid_output = False
    
    # Extract metadata from the line
    try:
        start = float(line.split()[0].split('-')[0])
        end = float(line.split()[0].split('-')[1])
        speaker = line.split()[1].replace(':', '')
    except (IndexError, ValueError):
        start, end, speaker = 0, 0, 'SPEAKER_00'
        
    while not valid_output and retry_count < max_retries:
        history.append({'role': 'user', 'content': f'{pre_prompt} {line}'})
        
        # Start timing
        start_time = time.time()
        
        response = client.chat(model=OLLAMA_MODEL, messages=history, options={'temperature': 0.5, 'num_ctx': OLLAMA_CONTEXT_SIZE})
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        token_count = response['eval_count'] + response['prompt_eval_count']
        print(f'token_count: {token_count}, duration: {duration:.2f} seconds')
        
        llm_response = response['message']['content']

        # Ensure the LLM response is properly formatted by stripping to the JSON content
        json_response = extract_json_from_text(llm_response)

        if json_response:
            llm_response = json_response
            actual_text_segment = extract_text_segment(line)
            corrected_response = correct_llm_output(llm_response, actual_text_segment, start, end, speaker)
            valid_output, error_message = validate_llm_output(corrected_response, actual_text_segment)
            if valid_output:
                history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
                llm_outputs.append(llm_response)  # Append the JSON object directly
                
                # Check if the token count exceeds and reset the context history if necessary
                if token_count > OLLAMA_CONTEXT_SIZE * 0.8:
                    history = initialize_history()
                    history.append({'role': 'user', 'content': f'{pre_prompt} {line}'})
                    history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
                    
            else:
                retry_count += 1
                print(f"Invalid format for response: {llm_response}")
                print(f"Error: {error_message}")
                history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
                history.append({'role': 'user', 'content': f"The previous response was invalid because: {error_message}. Please correct it."})
        else:
            retry_count += 1
            print(f"Response is not properly formatted JSON: {llm_response}")
            history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
            history.append({'role': 'user', 'content': "The previous response was not properly formatted JSON. Please correct it."})
            
    if not valid_output:
        print(f"Failed to get a valid response after {max_retries} attempts")
        fallback_response = create_fallback_response(line, start, end, speaker)
        llm_outputs.append(fallback_response)

    return history, llm_outputs


def extract_json_from_text(text):
    try:
        # Clean up the text by removing code block markers if present
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()

        # Use demjson3 to decode the JSON content, which handles malformed JSON gracefully
        parsed_json = demjson3.decode(text)
        
        # Validate that the required keys are present in the parsed JSON
        required_keys = {"text_segment", "fallacy_explanation", "fallacy_type", "speaker", "start", "end", "gif_query"}
        if not required_keys.issubset(parsed_json.keys()):
            raise ValueError(f"Missing required keys in JSON: {parsed_json.keys()}")
        
        return parsed_json

    except (demjson3.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during JSON extraction: {e}")
        return None

def correct_llm_output(response, text_segment, start, end, speaker):
    response['text_segment'] = text_segment
    response['start'] = start
    response['end'] = end
    response['speaker'] = speaker
    return response

def create_fallback_response(line, start, end, speaker):
    return {
        'text_segment': line,
        'fallacy_explanation': "NA",
        'fallacy_type': ["NA"],
        'speaker': speaker,
        'start': start,
        'end': end,
        'gif_query': ''
    }

def extract_text_segment(line):
    # Extract the text segment by splitting on the first colon and taking the second part
    parts = line.split(':', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return line.strip()

def validate_llm_output(output, input_text_segment):
    try:
        data = output
        required_fields = ["text_segment", "fallacy_explanation", "fallacy_type", "speaker", "start", "end", "gif_query"]
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"

        # Check if fallacy_type is a list
        if not isinstance(data["fallacy_type"], list):
            return False, "fallacy_type should be a list"

        # Validate data types of fields
        if not isinstance(data["text_segment"], str):
            return False, "text_segment should be a string"
        if not isinstance(data["fallacy_explanation"], str):
            return False, "fallacy_explanation should be a string"
        if not isinstance(data["speaker"], str):
            return False, "speaker should be a string"
        if not isinstance(data["start"], (int, float)):
            return False, "start should be a number"
        if not isinstance(data["end"], (int, float)):
            return False, "end should be a number"
        if not isinstance(data["gif_query"], str):
            return False, "giphy_search_query should be a string"

        # Check if text_segment matches the input text segment content
        if data["text_segment"].strip() != input_text_segment.strip():
            return False, "text_segment does not match the input text segment content"

        return True, ""
    except (json.JSONDecodeError, TypeError):
        return False, "Invalid JSON format"


def read_line_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def save_llm_output_to_json(llm_outputs, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(llm_outputs, json_file, indent=2)

def save_intermediate_results(temp_file, history, llm_outputs, processed_lines):
    temp_data = {
        'history': history,
        'llm_outputs': llm_outputs,
        'processed_lines': processed_lines
    }
    with open(temp_file, 'w') as f:
        json.dump(temp_data, f, indent=2)

