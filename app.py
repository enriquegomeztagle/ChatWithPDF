import streamlit as st
import requests
import re


# Helper function to format the response
def format_response(response_text):
    """
    Formats the response text, styling content inside <think> tags with smaller, lighter text,
    while keeping the rest of the text normal and maintaining line breaks.
    """
    # Using regex to capture everything inside <think>...</think>
    think_match = re.search(r"<think>(.*?)<\/think>", response_text, re.DOTALL)
    if think_match:
        # Extract <think> content and clean up
        think_content = think_match.group(1).strip()

        # Replace newlines in <think> content with <br> for proper HTML rendering
        think_content = think_content.replace("\n", "<br>")

        # Remove <think>...</think> from response text to get normal content
        normal_content = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        ).strip()

    else:
        # Default case if no <think> content is found
        think_content = ""
        normal_content = response_text.strip()

    # Format the content
    formatted_think = (
        (
            f"<p style='font-size: 0.75em; font-style: italic; color: rgba(255, 255, 255, 0.75); line-height: 1.5;'>{think_content}</p>"
        )
        if think_content
        else ""
    )

    formatted_normal = f"<p style='font-size: 1em; font-weight: normal; color: #ffffff;'>{normal_content}</p>"

    return formatted_think + formatted_normal


# Render additional details with the same formatting as <think>
def format_additional_details(details):
    """
    Formats the additional details in a horizontal layout with inline-block styling.
    """
    formatted_details = ""
    for key, value in details.items():
        formatted_details += (
            f"<div style='display: inline-block; margin-right: 2em; font-size: 0.75em; font-style: italic; color: rgba(255, 255, 255, 0.75);'>"
            f"<strong>{key}:</strong> {value}</div>"
        )
    return formatted_details


# Main function to run the Streamlit app
def main():
    """
    Main function to run the Streamlit app, allowing users to ask questions about their PDF documents
    and generate responses using the Ollama-based API.
    """
    st.set_page_config(page_title="📄 AI PDF Assistant", layout="wide")

    # Title and Description Section
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>
        📄 AI PDF Assistant
        </h1>
        <h3 style='text-align: center; color: #34495E;'>
        🤖 Unlock the power of your PDF documents with this AI-powered PDF Document Assistant 📁
        </h3>
        <p style='text-align: center; color: #7F8C8D;'>
        Seamlessly extract insights from your PDF documents using cutting-edge AI technology.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # User Interaction Section
    user_question = st.text_input(
        "🔍 Ask a question related to your PDF documents:",
        "",
        help="Type in a question to get insights from the loaded PDF documents.",
    )

    # Get Response Section
    if st.button("🤖 Get Response"):
        if user_question.strip():
            with st.spinner("Generating response from the API..."):
                # Send request to the FastAPI server
                api_url = "http://localhost:8000/query"
                payload = {"question": user_question}

                try:
                    response = requests.post(api_url, json=payload)
                    if response.status_code == 200:
                        api_response = response.json()

                        # Format the response
                        formatted_response = format_response(
                            api_response.get("response", "No response received.")
                        )

                        st.markdown("### 🤖 AI's Response")
                        st.markdown(formatted_response, unsafe_allow_html=True)

                        # Display additional details
                        st.markdown("###### 📜 Additional Details:")
                        details = {
                            "Model Name": api_response.get("model_name", "N/A"),
                            "Embeddings Name": api_response.get(
                                "embeddings_name", "N/A"
                            ),
                            "Query Time (seconds)": api_response.get(
                                "query_time_seconds", "N/A"
                            ),
                            "Request UUID": api_response.get("request_uuid", "N/A"),
                            "Timestamp": api_response.get("timestamp", "N/A"),
                        }

                        formatted_details = format_additional_details(details)
                        st.markdown(formatted_details, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.success("Response generated successfully!")

                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter a valid question before generating a response.")


if __name__ == "__main__":
    main()
