import string
import subprocess

from langchain.prompts import PromptTemplate


def append_query_with_acronym(user_input, configs, session_state):
    # Initialize the prompt template for acronym detection
    llm_acronym_check = PromptTemplate(
        input_variables=["user_query"],
        template=configs['prompts']['DETECT_ACRONYM_template']['template']
    )

    # Get the LLM response containing detected acronyms
    llm_response = session_state.llm(llm_acronym_check.format(user_query=user_input))
    # Split the response into individual acronyms and strip whitespace and punctuation
    detected_acronyms = [
        acro.strip(string.whitespace + string.punctuation)
        for acro in llm_response.split(',')
        if acro.strip()
    ]
    print(f"\nDetected Acronyms: {detected_acronyms} for the USER_INPUT: {user_input}")
    # Initialize a list to collect definitions
    definitions = []
    if detected_acronyms:
        # Create a case-insensitive mapping of LIGO_ACRONYMS
        ligo_acronyms_lower = {k.lower(): v for k, v in session_state.LIGO_ACRONYMS.items()}
        for acro in detected_acronyms:
            acro_lower = acro.lower()
            # Skip the acronym if it's 'ligo' (case-insensitive)
            if acro_lower != "ligo":
                # Retrieve the definition using the lowercase acronym
                acro_definition = ligo_acronyms_lower.get(acro_lower)

                if acro_definition:
                    # Append the definition to the user input with a period separator
                    user_input = f"{user_input}. {acro_definition}"
                    definitions.append(acro_definition)  # Collect the definition
                    print(f"Appended Query: {user_input}")
                else:
                    print(f"Definition for acronym '{acro}' not found.")
    # Concatenate all collected definitions into a single string separated by spaces
    concatenated_definitions = ' '.join(definitions)
    # Optionally, you can also log the concatenated definitions
    print(f"Concatenated Definitions: {concatenated_definitions}")
    # Return both the modified user_input and the concatenated definitions
    return user_input, concatenated_definitions


def get_chat_history_string(session_state, num_last_lines=6):
    chat_history_string_full = "\n".join([f"{msg['role']}: {msg['message']}" for msg in session_state.chat_history])
    # Split the input text into lines and filter
    chat_history_string_last_few_lines = [line for line in chat_history_string_full.splitlines(
    ) if line.startswith('user:') or line.startswith('assistant:')]
    chat_history_string_last_few_lines = chat_history_string_last_few_lines[-num_last_lines:-1]
    chat_history_string_last_few_lines = "\n".join(chat_history_string_last_few_lines)
    return chat_history_string_last_few_lines


def process_user_query_and_switch_states(configs, session_state):
    print('Executing process_user_query_and_switch_states')

    # Initialize Prompt Templates
    llm_generate_meaningful_query_prompt = PromptTemplate(
        input_variables=["history", "user_input"],
        template=configs['prompts']['user_query_template']['template']
    )

    llm_check_user_reaction_prompt = PromptTemplate(
        input_variables=["user_reaction"],
        template=configs['prompts']['user_reaction_template']['template']
    )

    llm_final_query_check = PromptTemplate(
        input_variables=["user_query"],
        template=configs['prompts']['query_content_template']['template']
    )

    # Initialize concatenated_definitions_list in session_state if not already present
    if 'concatenated_definitions_list' not in session_state:
        session_state.concatenated_definitions_list = []

    user_input = session_state.user_input

    # Handle empty or whitespace-only user input
    if not user_input.strip():
        llm_response = "Please enter a more specific query."
        session_state.chat_history.append({"role": "assistant", "message": llm_response})
        # Return default values when input is empty
        return None, ""

    # Handle different stages of the state machine
    if session_state.stage == 'initial':
        print("State: initial")
        # Retrieve the latest user input from chat history
        user_input = session_state.chat_history[-1]['message']

        # Detect and append acronym definitions
        user_input, concatenated_definitions_stage_0 = append_query_with_acronym(user_input, configs, session_state)
        if concatenated_definitions_stage_0:
            session_state.concatenated_definitions_list.append(concatenated_definitions_stage_0)

        # Generate a meaningful query using LLM
        llm_response = session_state.llm(
            llm_generate_meaningful_query_prompt.format(history=get_chat_history_string(session_state),
                                                        user_input=user_input)
        )

        session_state.chat_history.append({"role": "assistant", "message": llm_response})

        # Transition to 'user_reaction' stage
        session_state.stage = 'user_reaction'
        print("Current State: Initial")
        print(f"User Input: {user_input}")
        print(f"Updated Query by LLM: {llm_response}")
        print("Switching state to: user_reaction")

        # Return to allow for the next state processing
        return None, ' '.join(session_state.concatenated_definitions_list).strip()

    elif session_state.stage == 'user_reaction':
        print("State: user_reaction")
        # Retrieve the latest message from chat history
        user_input = session_state.chat_history[-1]['message']

        # Detect and append any new acronym definitions
        user_input, concatenated_definitions_stage_1 = append_query_with_acronym(user_input,configs,session_state)
        if concatenated_definitions_stage_1:
            session_state.concatenated_definitions_list.append(concatenated_definitions_stage_1)

        # Check user's reaction using LLM
        llm_check_user_reaction = session_state.llm(
            llm_check_user_reaction_prompt.format(user_reaction=user_input)
        )

        if "true" in llm_check_user_reaction.strip().lower():
            print("User confirmed query. Doing final check before proceeding with RAG.")

            # Retrieve the confirmed query from chat history
            confirmed_query = session_state.chat_history[-1]['message']
            print(f"Confirmed Query : {confirmed_query}")

            # [Optional] Final query validation can be added here
            most_common = True  # Assuming confirmation for simplicity

            if most_common:
                # st.markdown(
                #     "I understand what you're looking for. Let me search through the resources and find the answer.")
                session_state.stage = 'initial'

                # Concatenate all collected definitions
                concatenated_definitions = ' '.join(session_state.concatenated_definitions_list).strip()

                # Reset the definitions list for future queries
                session_state.concatenated_definitions_list = []

                # Return the confirmed query and concatenated definitions
                return confirmed_query, concatenated_definitions
            else:
                print("User question seems to be invalid. Asking for more clarification.")
                assistant_response = "I'm sorry, I am still trying to understand your question. Can you explain a bit more?"
                # st.markdown(assistant_response)
                session_state.chat_history.append({"role": "assistant", "message": assistant_response})
                session_state.stage = 'initial'

                # Concatenate and reset definitions
                concatenated_definitions = ' '.join(session_state.concatenated_definitions_list).strip()
                session_state.concatenated_definitions_list = []

                # Return default values when query is invalid
                return None, concatenated_definitions

        elif "false" in llm_check_user_reaction.strip().lower():
            print("User did not confirm. Asking for more clarification.")
            assistant_response = "I'm sorry, I am still trying to understand your question. Can you explain a bit more?"
            # st.markdown(assistant_response)
            session_state.chat_history.append({"role": "assistant", "message": assistant_response})
            session_state.stage = 'initial'

            # Concatenate and reset definitions
            concatenated_definitions = ' '.join(session_state.concatenated_definitions_list).strip()
            session_state.concatenated_definitions_list = []

            # Return default values when user does not confirm
            return None, concatenated_definitions

        else:
            print("User seems confused. Handling confusion by asking for clarification.")
            session_state.stage = 'confused'
            print("State: User-Confused")

            # Retrieve the latest message from chat history
            user_input = session_state.chat_history[-1]['message']

            # Detect and append any new acronym definitions
            user_input, concatenated_definitions_stage_2 = append_query_with_acronym(user_input,configs,session_state)
            if concatenated_definitions_stage_2:
                session_state.concatenated_definitions_list.append(concatenated_definitions_stage_2)

            # Generate a meaningful query using LLM
            llm_response = session_state.llm(
                llm_generate_meaningful_query_prompt.format(history=get_chat_history_string(session_state),
                                                            user_input=user_input)
            )
            # st.markdown(f"Updated Query: {llm_response}")
            session_state.chat_history.append({"role": "assistant", "message": llm_response})

            # Transition back to 'user_reaction' stage
            session_state.stage = 'user_reaction'
            print("Current State: User-Confused")
            print(f"User Input: {user_input}")
            print(f"Updated Query by LLM: {llm_response}")
            print("Switching state to: user_reaction")

            # Concatenate definitions without returning since confirmation isn't achieved yet
            concatenated_definitions = ' '.join(session_state.concatenated_definitions_list).strip()
            return None, concatenated_definitions

    else:
        print("Invalid state. Resetting to initial.")
        session_state.stage = 'initial'

        # Concatenate and reset definitions
        concatenated_definitions = ' '.join(session_state.concatenated_definitions_list).strip()
        session_state.concatenated_definitions_list = []

        # Return default values for invalid state
        return None, concatenated_definitions



def get_chat_history_string(session_state, num_last_lines=6):
    chat_history_string_full = "\n".join([f"{msg['role']}: {msg['message']}" for msg in session_state.chat_history])
    # Split the input text into lines and filter
    chat_history_string_last_few_lines = [line for line in chat_history_string_full.splitlines(
    ) if line.startswith('user:') or line.startswith('assistant:')]
    chat_history_string_last_few_lines = chat_history_string_last_few_lines[-num_last_lines:-1]
    chat_history_string_last_few_lines = "\n".join(chat_history_string_last_few_lines)
    return chat_history_string_last_few_lines
