DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


OA_PROMPT_DICT = (
    "<|prompter|> I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} tone of voice. Recipe: {recipe} This is the current step: {current_step}. <|endoftext|> <|assistant|> ok! <|endoftext|> {dialog} <|endoftext|> <|assistant|>"
)

VICUNA_PROMPT_DICT = (
    "{user_token} I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} "
    "tone of voice. Recipe: {recipe} {current_step} {sep_token} {sys_token} ok! {sep_token} {dialog} {sys_token} "
)

CURRENT_STEP_TEMP = "We are on Step {step_num}: {step_text}"
NO_STEP_TEMP = "We are just starting the recipe"



SEP_TOKEN = "\n"
IMG_TOKEN = "[IMG]"
IMG_START_TOKEN = "<im_start>"
IMG_END_TOKEN = "<im_end>"
RET_TOKEN = "[RET]"
RET_TOKEN_2 = "[RET2]"

IGNORE_INDEX = -100


CONTEXT_WINDOW = 3

# ======== DEFAULT ARGUMENTS ========
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_STOP_CRITERIA = -1.0
DEFAULT_SEED = 11731
DEFAULT_LR_STEP_SIZE = 250
DEFAULT_LR_NUM_CYCLES = 10
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_LEN = 512
DEFAULT_GRAD_CLIP = 0.5


# ======== DEFAULT PROMPTS ========

CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['from'] != 'assistant' %}{{ '<|im_start|>system\nYou are a helpful AI conversational assistant. You can retrieve images or video moments by generating the [RET] token.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['from'] + '\n' + message['value'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

USER_CAPTION_REQUEST_TEMPLATES = [
    "Please describe the image.",
    "What is happening in the image?",
    "What can you see in the image?",
    "Describe the image.",
    "What is in the image?",
    "What is the image about?",
    "What is the image showing?",
    "Please describe the image.",
    "What is happening in the image?",
    "What can you see in the image?",
]

USER_RETRIEVAL_REQUEST_TEMPLATES = [
    "Please find an image with this description: ",
    "Find an image that matches this description: ",
    "I am looking for an image that looks like this: ",
    "Can you find an image that matches this description: ",
    "Please find an image that could be described like this: ",
    "Find an image that looks like this: ",
    "I am looking for an image that could be described like this: ",
]

SYSTEM_RETRIEVAL_RESPONSE_TEMPLATES = [
    "Here is an image that matches your description.",
    "I found an image that matches your description.",
    "This image matches your description.",
    "Here is an image that could be described like that.",
    "I found an image that could be described like that.",
    "This image could be described like that.",
    "Here is an image that matches your description.",
    "I found an image that matches your description.",
]