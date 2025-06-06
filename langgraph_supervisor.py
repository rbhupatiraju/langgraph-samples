from typing import Dict, TypedDict, Annotated, Sequence, Literal
from langgraph.graph import Graph, StateGraph
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator

# Define the state that will be passed between nodes
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    data: str
    solution: str
    next_step: Literal["data_agent", "problem_solver", "end"]

# Create the data agent
def create_data_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data gathering agent. Your job is to collect and organize relevant data based on the user's request."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    
    def data_agent(state: AgentState) -> AgentState:
        print("\n=== Data Agent State ===")
        print(f"Current messages: {state['messages']}")
        response = chain.invoke({"messages": state["messages"]})
        print(f"Data gathered: {response.content[:100]}...")  # Print first 100 chars
        return {
            "data": response.content,
            "next_step": "supervisor"  # Changed to go back to supervisor
        }
    
    return data_agent

# Create the problem solver agent
def create_problem_solver():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a problem solver agent. Your job is to analyze the data and provide solutions to the user's problem."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Here is the data you need to analyze: {data}"),
    ])
    
    chain = prompt | llm
    
    def problem_solver(state: AgentState) -> AgentState:
        print("\n=== Problem Solver State ===")
        print(f"Current messages: {state['messages']}")
        print(f"Data to analyze: {state['data'][:100]}...")  # Print first 100 chars
        response = chain.invoke({
            "messages": state["messages"],
            "data": state["data"]
        })
        print(f"Solution provided: {response.content[:100]}...")  # Print first 100 chars
        return {
            "solution": response.content,
            "next_step": "supervisor"
        }
    
    return problem_solver

# Create the supervisor agent
def create_supervisor_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor agent. Your job is to:
1. Analyze the user's request
2. Determine if you need more data or can proceed with problem solving
3. Coordinate the workflow between data gathering and problem solving
4. Provide final responses to the user

If the user's request requires data gathering, respond with "data_agent"
If the user's request can be solved with existing data, respond with "problem_solver"
If the task is complete, respond with "end"

Only respond with one of these three options: "data_agent", "problem_solver", or "end"
"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    
    def supervisor(state: AgentState) -> AgentState:
        print("\n=== Supervisor State ===")
        print(f"Current messages: {state['messages']}")
        print(f"Current data: {state.get('data', '')[:100]}...")  # Print first 100 chars
        print(f"Current solution: {state.get('solution', '')[:100]}...")  # Print first 100 chars
        
        # If we have a solution, provide the final response
        if state.get("solution"):
            print("Solution found, providing final response")
            # Return the solution directly as the final response
            return {
                "messages": [AIMessage(content=state["solution"])],
                "next_step": "end"
            }
        
        # If we have data but no solution, go to problem solver
        if state.get("data") and not state.get("solution"):
            print("Data available but no solution, routing to problem solver")
            return {"next_step": "problem_solver"}
        
        # Otherwise, determine the next step
        print("Determining next step based on user request")
        response = chain.invoke({"messages": state["messages"]})
        next_step = response.content.strip().lower()
        print(f"Decided next step: {next_step}")
        
        # Validate the response
        if next_step not in ["data_agent", "problem_solver", "end"]:
            print(f"Invalid response, defaulting to data_agent")
            next_step = "data_agent"
            
        return {"next_step": next_step}
    
    return supervisor

# Create the workflow
def create_workflow():
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", create_supervisor_agent())
    workflow.add_node("data_agent", create_data_agent())
    workflow.add_node("problem_solver", create_problem_solver())
    
    # Define the edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {
            "data_agent": "data_agent",
            "problem_solver": "problem_solver",
            "end": END  # Use None to indicate end of workflow
        }
    )
    
    workflow.add_edge("data_agent", "supervisor")
    workflow.add_edge("problem_solver", "supervisor")
    
    # Set the entry point
    workflow.set_entry_point("supervisor")
    
    # Set the end condition
    workflow.set_finish_point("supervisor")  # Supervisor will handle the end condition
    
    # Compile the graph
    return workflow.compile()

# Example usage
def invoke_workflow(message: str):
    # Initialize the workflow
    workflow = create_workflow()
    
    # Example input
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "data": "",
        "solution": "",
        "next_step": "supervisor"
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Print the final response
    if result["messages"]:
        print("\n=== Final Result ===")
        print("Response:", result["messages"][-1].content)
    else:
        print("No response generated")

invoke_workflow("How many people live in Mumbai? How many people live in Tokyo? What is the sum of the two populations?")
