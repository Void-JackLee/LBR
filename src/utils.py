def get_prompt(dataset_name):
    if "Toy" in dataset_name:
        instruction_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
        history_prompt = "The user has played the following toys before: "
    elif ("Movie" in dataset_name) or ("ml-" in dataset_name):
        instruction_prompt = "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user."
        history_prompt = "The user has watched the following movies before: "
    elif "Yelp" in dataset_name:
        instruction_prompt = "Given a list of restaurants the user has visted before, please recommend a new restaurant that the user likes to the user."
        history_prompt = "The user has visted the following restaurants before: "
    elif "Food" in dataset_name:
        instruction_prompt = "Given a list of recipes the user has cooked before, please recommend a new recipe that the user likes to the user."
        history_prompt = "The user has cooked the following recipes before: "
    elif "Book" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "Clothing" in dataset_name:
        instruction_prompt = "Given a list of clothing the user has worn before, please recommend a new clothing that the user likes to the user."
        history_prompt = "The user has worn the following clothing before: "
    elif "Game" in dataset_name:
        instruction_prompt = "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user."
        history_prompt = "The user has played the following video games before: "
    elif "Office" in dataset_name:
        instruction_prompt = "Given a list of office products the user has used before, please recommend a new office product that the user likes to the user."
        history_prompt = "The user has used the following office products before: "
    elif "Beauty" in dataset_name:
        instruction_prompt = "Given a list of beauty products the user has used before, please recommend a new beauty product that the user likes to the user."
        history_prompt = "The user has used the following beauty products before: "
    elif "Music" in dataset_name:
        instruction_prompt = "Given a list of music the user has listened before, please recommend a new music that the user likes to the user."
        history_prompt = "The user has listened the following music before: "
    elif "Electronic" in dataset_name:
        instruction_prompt = "Given a list of electronic products the user has used before, please recommend a new electronic product that the user likes to the user."
        history_prompt = "The user has used the following electronic products before: "
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return instruction_prompt, history_prompt

def generate_prompt(data_point, template=False):
    if data_point["input"]:
        input = f'{data_point["input_prefix_str"]}{", ".join("{}" for _ in data_point["input_arr"])}\n ' if template else data_point["input"]
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{input}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    
def generate_prompt_before_items(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input_prefix_str"][:-1]}
"""