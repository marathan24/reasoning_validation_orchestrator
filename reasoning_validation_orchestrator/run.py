import logging
import os
import json
import asyncio
import traceback
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

from naptha_sdk.modules.orchestrator import Orchestrator
from naptha_sdk.modules.agent import Agent
from naptha_sdk.schemas import OrchestratorRunInput, AgentRunInput
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from reasoning_validation_orchestrator.schemas import InputSchema

logger = logging.getLogger(__name__)

# Define Pydantic model specifically for validation agent input
class ValidationInputModel(BaseModel):
    func_name: str
    problem: str
    thoughts: List[str]

class ReasoningValidationOrchestrator(Orchestrator):
    async def create(self, deployment, *args, **kwargs):
        """Initialize the orchestrator with agent deployments."""
        logger.info(f"Creating orchestrator with deployment: {deployment}")
        
        self.deployment = deployment
        # Ensure we have the necessary agent deployments
        if not hasattr(deployment, 'agent_deployments') or len(deployment.agent_deployments) < 2:
            logger.error(f"Missing or insufficient agent deployments. Found: {getattr(deployment, 'agent_deployments', None)}")
            try:
                config_path = "reasoning_validation_orchestrator/configs/deployment.json"
                logger.info(f"Attempting to load agent deployments from {config_path}")
                
                with open(config_path, "r") as f:
                    deployments_config = json.load(f)
                
                if isinstance(deployments_config, list) and len(deployments_config) > 0:
                    deployment_config = deployments_config[0]
                    
                    if "agent_deployments" in deployment_config and len(deployment_config["agent_deployments"]) >= 2:
                        deployment.agent_deployments = deployment_config["agent_deployments"]
                        logger.info(f"Successfully loaded {len(deployment.agent_deployments)} agent deployments")
                    else:
                        raise ValueError("Configuration doesn't have enough agent deployments")
                else:
                    raise ValueError("Invalid deployment configuration format")
                
            except Exception as e:
                logger.error(f"Failed to load agent deployments from configuration: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError("Orchestrator requires exactly 2 agent deployments: reasoning and validation")
        
        if not hasattr(deployment, 'agent_deployments') or len(deployment.agent_deployments) < 2:
            raise ValueError("Orchestrator requires exactly 2 agent deployments: reasoning and validation")
        
        self.reasoning_deployment = deployment.agent_deployments[0]
        self.validation_deployment = deployment.agent_deployments[1]
        
        # Get private key for agent authentication
        private_key_path = os.getenv("PRIVATE_KEY_FULL_PATH", os.getenv("PRIVATE_KEY"))
        if not private_key_path:
            raise ValueError("Private key path environment variable (PRIVATE_KEY_FULL_PATH or PRIVATE_KEY) not set.")
        self.private_key = get_private_key_from_pem(private_key_path)

    async def run_agent(self, agent_deployment, inputs, consumer_id):
        """
        Run an agent with the provided inputs and handle the response.
        This method encapsulates all the complexity of agent execution.
        """
        agent = Agent()
        await agent.create(deployment=agent_deployment)
        
        agent_run_input = AgentRunInput(
            consumer_id=consumer_id,
            inputs=inputs,  # Pass inputs directly without conversion
            deployment=agent_deployment,
            signature=sign_consumer_id(consumer_id, self.private_key)
        )
        
        result = await agent.run(agent_run_input)
        
        if result.status == 'error':
            logger.error(f"Agent run failed: {result.error_message}")
            raise RuntimeError(f"Agent failed: {result.error_message}")
        
        # Extract results from the agent response
        if hasattr(result, 'results') and result.results:
            try:
                if isinstance(result.results[0], str):
                    return json.loads(result.results[0])
                elif isinstance(result.results[0], dict):
                    return result.results[0]
                else:
                    return result.results[0]
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract results: {e}")
                logger.error(f"Results: {result.results}")
                raise ValueError(f"Invalid results format: {e}")
        else:
            logger.warning("Agent result does not have expected structure or is empty")
            return {}

    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        """
        Execute the reasoning and validation workflow.
        """
        try:
            # Step 1: Run the reasoning agent to generate thoughts
            logger.info("Starting reasoning agent")
            
            # Clean the problem string - remove trailing commas or other problematic characters
            problem = module_run.inputs.problem.rstrip(',').strip()
            
            # Keep the reasoning input as a dictionary (no change from original)
            reasoning_input = {
                "func_name": "reason",
                "problem": problem,
                "num_thoughts": module_run.inputs.num_thoughts
            }
            
            reasoning_result = await self.run_agent(
                self.reasoning_deployment,
                reasoning_input,  # Dictionary input for reasoning agent
                module_run.consumer_id
            )
            
            # Extract thoughts from the reasoning result
            thoughts = reasoning_result.get('thoughts', [])
            logger.info(f"Reasoning completed with {len(thoughts)} thoughts")
            
            # Clean the thoughts to ensure they are all properly formatted strings
            clean_thoughts = []
            for thought in thoughts:
                if isinstance(thought, dict) and 'content' in thought:
                    clean_thoughts.append(thought['content'])
                elif isinstance(thought, str):
                    clean_thoughts.append(thought)
                else:
                    clean_thoughts.append(str(thought))
            
            logger.info(f"Prepared {len(clean_thoughts)} thoughts for validation")
            
            # Step 2: Use a Pydantic model for validation agent
            validation_input = ValidationInputModel(
                func_name="validate",
                problem=problem,
                thoughts=clean_thoughts
            )
            
            validation_result = await self.run_agent(
                self.validation_deployment,
                validation_input,  # Pydantic model for validation agent
                module_run.consumer_id
            )
            
            # Extract the best thought and final answer
            best_thought_idx = validation_result.get('best_thought_index', 0)
            final_answer = validation_result.get('final_answer', '')
            
            best_thought = ""
            if thoughts and 0 <= best_thought_idx < len(thoughts):
                best_thought = thoughts[best_thought_idx]
            elif thoughts:
                logger.warning(f"Best thought index {best_thought_idx} invalid for {len(thoughts)} thoughts. Using index 0.")
                best_thought = thoughts[0]
            else:
                logger.warning("No thoughts available to determine best thought.")
            
            logger.info(f"Validation completed, selected best thought index: {best_thought_idx}")
            
            # Return the final result
            return {
                "status": "success",
                "problem": problem,
                "reasoning_thoughts": thoughts,
                "validation_result": validation_result,
                "final_answer": final_answer,
                "best_thought": best_thought
            }
            
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Orchestrator execution failed: {str(e)}")

async def run(module_run: dict, *args, **kwargs):
    """
    Entry point for the orchestrator module.
    """
    try:
        # Convert the input to an OrchestratorRunInput object
        module_run_obj = OrchestratorRunInput(**module_run)
        
        # Validate the inputs against the schema
        try:
            module_run_obj.inputs = InputSchema(**module_run_obj.inputs)
        except Exception as e:
            logger.error(f"Failed to parse orchestrator inputs against InputSchema: {e}")
            raise ValueError(f"Invalid input format: {e}")
        
        # Create and run the orchestrator
        orchestrator = ReasoningValidationOrchestrator()
        await orchestrator.create(module_run_obj.deployment, *args, **kwargs)
        result = await orchestrator.run(module_run_obj, *args, **kwargs)
        return result
        
    except Exception as e:
        logger.error(f"Error in orchestrator run: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Test the orchestrator locally
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    
    naptha = Naptha()
    
    deployment = asyncio.run(setup_module_deployment(
        "orchestrator",
        "reasoning_validation_orchestrator/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))
    
    input_params = {
        "problem": "What is the sum of all integers from 1 to 100?",
        "num_thoughts": 3
    }
    
    deployment_dict = deployment.dict(exclude_unset=True) if hasattr(deployment, 'dict') else deployment
    
    module_run = {
        "inputs": input_params,
        "deployment": deployment_dict,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
    }
    
    response = asyncio.run(run(module_run))
    print("Final Answer:", response.get("final_answer"))
    print("Best Reasoning Path:", response.get("best_thought", ""))