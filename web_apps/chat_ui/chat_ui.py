import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

st.set_page_config(page_title="Model Comparison Chatbot", page_icon="🤖")

# Model configurations
models = {
    "Original Model": {
        "name": "microsoft/phi-2",
        "peft": False
    },
    "Fine-tuned Model": {
        "name": "microsoft/phi-2",
        "peft_path": "outputs/finetuned-model",
        "peft": True
    }
}

def load_model(model_choice):
    """Load the selected model"""
    if model_choice not in st.session_state.loaded_models:
        config = models[model_choice]

        with st.spinner(f"Loading {model_choice}..."):
            tokenizer = AutoTokenizer.from_pretrained(config["name"])
            base_model = AutoModelForCausalLM.from_pretrained(
                config["name"],
                torch_dtype=torch.float16,
                device_map="auto"
            )

            if config["peft"]:
                model = PeftModel.from_pretrained(base_model, config["peft_path"])
            else:
                model = base_model

            st.session_state.loaded_models[model_choice] = {
                "tokenizer": tokenizer,
                "model": model
            }

    return st.session_state.loaded_models[model_choice]

def generate_response(tokenizer, model, user_input):
    """Generate response from the model"""
    system_prompt = "You are a helpful AI assistant. Provide clear, concise answers.\n"

    full_prompt = system_prompt + "Human: " + user_input + "\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            min_length=inputs.input_ids.shape[1] + 10,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    response_start = full_response.find("Assistant:")
    if response_start != -1:
        response = full_response[response_start + len("Assistant:"):].strip()
    else:
        response = full_response[len(full_prompt):].strip()

    return response

# Initialize session state
if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# UI Layout
st.title("🤖 Model Comparison Chatbot")

# Model selector in sidebar
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=0,
        help="Choose between the original and fine-tuned models for comparison"
    )

    if st.button("Load Model"):
        if selected_model != st.session_state.current_model:
            with st.spinner(f"Loading {selected_model}..."):
                components = load_model(selected_model)
                st.session_state.current_model = selected_model
                st.session_state.messages = []  # Clear chat when switching models
            st.success(f"{selected_model} loaded successfully!")
        else:
            st.info(f"{selected_model} is already loaded.")

# Display current model
if st.session_state.current_model:
    st.info(f"Currently using: **{st.session_state.current_model}**")
else:
    st.warning("Please load a model first.")

# Chat interface
st.header("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(
    "Ask me anything...",
    disabled=st.session_state.current_model is None
):
    if not st.session_state.current_model:
        st.error("Please load a model first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        components = st.session_state.loaded_models[st.session_state.current_model]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(components["tokenizer"], components["model"], prompt)

            st.markdown(response)

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")
st.caption("Built with Streamlit, Transformers, and PEFT. Fine-tuned on coding tasks.")
