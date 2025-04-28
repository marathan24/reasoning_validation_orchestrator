import logging
import os
import json
import asyncio
import traceback
import uuid
import re
from dotenv import load_dotenv

from naptha_sdk.modules.agent import Agent
from naptha_sdk.schemas import OrchestratorRunInput, AgentRunInput, AgentDeployment
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from reasoning_validation_orchestrator.schemas import InputSchema

logger = logging.getLogger(__name__)

class ReasoningValidationOrchestrator:
    async def create(self, deployment, *args, **kwargs):
        logger.info(f"Creating orchestrator with deployment: {deployment}")

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
                        agent_deployments = []
                        for agent_config in deployment_config["agent_deployments"][:2]:
                            agent_config["node"] = deployment.node.dict() if hasattr(deployment.node, 'dict') else deployment.node
                            agent_deployments.append(AgentDeployment(**agent_config))

                        deployment.agent_deployments = agent_deployments
                        logger.info(f"Successfully loaded {len(agent_deployments)} agent deployments")
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

        self.deployment = deployment
        self.agent_deployments = deployment.agent_deployments

        try:
            self.reasoning_agent = Agent()
            await self.reasoning_agent.create(deployment=self.agent_deployments[0], *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize reasoning agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[0]}")
            raise

        try:
            self.validation_agent = Agent()
            await self.validation_agent.create(deployment=self.agent_deployments[1], *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize validation agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[1]}")
            raise
        
    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        run_id = str(uuid.uuid4())
        private_key_path = os.getenv("PRIVATE_KEY_FULL_PATH")
        if not private_key_path:
             logger.warning("PRIVATE_KEY_FULL_PATH environment variable not set. Trying PRIVATE_KEY.")
             private_key_path = os.getenv("PRIVATE_KEY")
             if not private_key_path:
                  raise ValueError("Private key path environment variable (PRIVATE_KEY_FULL_PATH or PRIVATE_KEY) not set.")

        private_key = get_private_key_from_pem(private_key_path)

        reasoning_input = {
            "func_name": "reason",
            "problem": module_run.inputs.problem,
            "num_thoughts": module_run.inputs.num_thoughts
        }

        reasoning_run_input = AgentRunInput(
            consumer_id=module_run.consumer_id,
            inputs=reasoning_input,
            deployment=self.agent_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, private_key)
        )

        try:
            reasoning_result = await self.reasoning_agent.run(reasoning_run_input)

            thoughts = []

            if hasattr(reasoning_result, 'results') and reasoning_result.results:
                try:
                    reasoning_data = json.loads(reasoning_result.results[0])
                    thoughts = reasoning_data.get('thoughts', [])
                    logger.info(f"Successfully extracted {len(thoughts)} thoughts from reasoning_result")
                    logger.debug(f"Raw thoughts: {thoughts}")
                except (json.JSONDecodeError, IndexError, TypeError) as e:
                    logger.error(f"Failed to extract thoughts from results: {e}")
                    logger.error(f"Results: {reasoning_result.results}")
            elif reasoning_result.status == 'error':
                 logger.error(f"Reasoning agent run failed: {reasoning_result.error_message}")
                 raise RuntimeError(f"Reasoning agent failed: {reasoning_result.error_message}")
            else:
                logger.warning(f"reasoning_result does not have expected structure or is empty: {reasoning_result}")

            logger.info(f"Reasoning completed with {len(thoughts)} thoughts")

        except Exception as e:
            logger.error(f"Error in reasoning step: {e}")
            if not isinstance(e, RuntimeError):
                 logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise e

        # Clean any potentially problematic characters from the problem string
        problem = module_run.inputs.problem
        if isinstance(problem, str):
            # Remove trailing commas, extra whitespace and other problematic characters
            problem = problem.rstrip(',').strip()

        # The validation agent expects each thought to be a simple string in the array
        # Make sure thoughts are in the right format
        clean_thoughts = []
        for thought in thoughts:
            if isinstance(thought, dict) and 'content' in thought:
                # Handle case where thoughts might be in object format
                clean_thoughts.append(thought['content'])
            elif isinstance(thought, str):
                # Normal case - just use the string directly
                clean_thoughts.append(thought)
            else:
                # Try to convert to string
                clean_thoughts.append(str(thought))
        
        logger.info(f"Prepared {len(clean_thoughts)} thoughts for validation")

        # Fix: Use a dictionary for inputs directly instead of JSON serializing it
        validation_input = {
            "func_name": "validate",
            "problem": problem,
            "thoughts": clean_thoughts
        }

        try:
            logger.debug(f"Validation input format: {type(validation_run_input.inputs)}")
            logger.debug(f"Validation thoughts type: {type(validation_run_input.inputs.get('thoughts', []))}")
            # Create the run input structure directly without any JSON serialization
            validation_run_input = AgentRunInput(
                consumer_id=module_run.consumer_id,
                inputs=validation_input,  # Pass the dictionary directly
                deployment=self.agent_deployments[1],
                signature=sign_consumer_id(module_run.consumer_id, private_key)
            )
            
            validation_result = await self.validation_agent.run(validation_run_input)
            
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise RuntimeError(f"Validation agent failed: {str(e)}")

        validation_data = {}

        if hasattr(validation_result, 'results') and validation_result.results:
            try:
                if isinstance(validation_result.results[0], str):
                    validation_data = json.loads(validation_result.results[0])
                elif isinstance(validation_result.results[0], dict):
                    validation_data = validation_result.results[0]
                logger.info(f"Successfully extracted validation data")
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract data from validation results: {e}")
                logger.error(f"Results: {validation_result.results}")
        elif validation_result.status == 'error':
             logger.error(f"Validation agent run failed: {validation_result.error_message}")
             raise RuntimeError(f"Validation agent failed: {validation_result.error_message}")
        else:
            logger.warning(f"validation_result does not have expected structure or is empty: {validation_result}")

        best_thought_idx = validation_data.get('best_thought_index', 0)
        final_answer = validation_data.get('final_answer', '')

        best_thought = ""
        if thoughts and 0 <= best_thought_idx < len(thoughts):
             best_thought = thoughts[best_thought_idx]
        elif thoughts:
             logger.warning(f"Best thought index {best_thought_idx} invalid for {len(thoughts)} thoughts. Using index 0.")
             best_thought = thoughts[0]
        else:
             logger.warning("No thoughts available to determine best thought.")

        logger.info(f"Validation completed, selected best thought index: {best_thought_idx}")

        result = {
            "problem": module_run.inputs.problem,
            "reasoning_thoughts": thoughts,
            "validation_result": validation_data,
            "final_answer": final_answer,
            "best_thought": best_thought
        }
        return result

async def run(module_run: dict, *args, **kwargs):
    module_run_obj = OrchestratorRunInput(**module_run)
    try:
        module_run_obj.inputs = InputSchema(**module_run_obj.inputs)
    except Exception as e:
         logger.error(f"Failed to parse orchestrator inputs against InputSchema: {e}")
         raise ValueError(f"Invalid input format: {e}")

    orchestrator = ReasoningValidationOrchestrator()
    await orchestrator.create(module_run_obj.deployment, *args, **kwargs)
    result = await orchestrator.run(module_run_obj, *args, **kwargs)
    return result

if __name__ == "__main__":
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