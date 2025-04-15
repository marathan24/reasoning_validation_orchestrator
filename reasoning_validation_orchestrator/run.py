import logging
import os
import json 
import asyncio
import traceback
import uuid
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
                config_path = os.path.join(os.path.dirname(__file__), "configs/agent_deployments.json") 
                logger.info(f"Attempting to load agent deployments from {config_path}")

                with open(config_path, "r") as f:
                    agent_deployments_config = json.load(f)

                orchestrator_config_path = os.path.join(os.path.dirname(__file__), "configs/deployment.json")
                with open(orchestrator_config_path, "r") as f:
                    orchestrator_deployments = json.load(f)

                current_orchestrator_config = next((d for d in orchestrator_deployments if d.get("name") == deployment.name), None)

                if not current_orchestrator_config:
                     raise ValueError(f"Could not find orchestrator configuration for deployment name: {deployment.name}")

                required_agent_names = [ad['name'] for ad in current_orchestrator_config.get("agent_deployments", [])]

                if len(required_agent_names) < 2:
                     raise ValueError("Orchestrator deployment config requires at least 2 agent_deployments listed.")

                deployment.agent_deployments = []
                found_agents = {}
                for agent_conf in agent_deployments_config:
                     if agent_conf.get("name") in required_agent_names:
                          if "node" not in agent_conf:
                               agent_conf["node"] = deployment.node.dict() if hasattr(deployment.node, 'dict') else deployment.node
                          if "module" not in agent_conf:
                               if agent_conf["name"] == required_agent_names[0]: agent_conf["module"] = {"name": "reasoning_modules"}
                               if agent_conf["name"] == required_agent_names[1]: agent_conf["module"] = {"name": "validation_modules"}
                          try:
                              agent_deployment_obj = AgentDeployment(**agent_conf)
                              found_agents[agent_conf["name"]] = agent_deployment_obj
                          except Exception as e:
                              logger.error(f"Error creating AgentDeployment for {agent_conf.get('name')}: {e}. Config: {agent_conf}")
                              raise

                for name in required_agent_names:
                    if name in found_agents:
                        deployment.agent_deployments.append(found_agents[name])
                    else:
                        raise ValueError(f"Could not find agent deployment configuration for required agent: {name}")

                if len(deployment.agent_deployments) < 2:
                     raise ValueError("Failed to load the required 2 agent deployments (reasoning, validation).")

                logger.info(f"Successfully loaded {len(deployment.agent_deployments)} agent deployments: {[ad.name for ad in deployment.agent_deployments]}")

            except FileNotFoundError as e:
                logger.error(f"Configuration file not found: {e}")
                raise ValueError(f"Configuration file missing: {e.filename}")
            except ValueError as e:
                logger.error(f"Configuration error: {e}")
                raise e
            except Exception as e:
                logger.error(f"Failed to load agent deployments from configuration: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError("Error processing agent deployment configurations.")



        if not hasattr(deployment, 'agent_deployments') or len(deployment.agent_deployments) < 2:
             raise ValueError("Orchestrator requires exactly 2 agent deployments after loading: reasoning and validation")

        self.deployment = deployment
        self.agent_deployments = deployment.agent_deployments


        if len(self.agent_deployments) != 2:
            raise ValueError(f"Expected 2 agent deployments, but found {len(self.agent_deployments)}: {[ad.name for ad in self.agent_deployments]}")

        try:
            self.reasoning_agent = Agent()
            
            await self.reasoning_agent.create(deployment=self.agent_deployments[0], *args, **kwargs)
            logger.info(f"Initialized reasoning agent with deployment: {self.agent_deployments[0].name}")
        except Exception as e:
            logger.error(f"Failed to initialize reasoning agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[0]}")
            raise

        try:
            self.validation_agent = Agent()
            await self.validation_agent.create(deployment=self.agent_deployments[1], *args, **kwargs)
            logger.info(f"Initialized validation agent with deployment: {self.agent_deployments[1].name}")
        except Exception as e:
            logger.error(f"Failed to initialize validation agent: {e}")
            logger.error(f"Agent deployment details: {self.agent_deployments[1]}")
            raise

    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        run_id = str(uuid.uuid4())
        private_key_pem = os.getenv("PRIVATE_KEY")
        if not private_key_pem:
             raise ValueError("PRIVATE_KEY environment variable not set.")
        try:
             private_key = get_private_key_from_pem(private_key_pem)
        except Exception as e:
             logger.error(f"Failed to load private key: {e}")
             raise ValueError("Invalid private key format or value.")


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

        reasoning_output = None 
        try:
            reasoning_agent_run_result = await self.reasoning_agent.run(reasoning_run_input)
            logger.info(f"Raw reasoning agent run result object type: {type(reasoning_agent_run_result)}")

            if hasattr(reasoning_agent_run_result, 'results') and reasoning_agent_run_result.results:
                result_json_string = reasoning_agent_run_result.results[0]
                reasoning_output = json.loads(result_json_string)
                logger.info(f"Reasoning completed with {len(reasoning_output['thoughts'])} thoughts")
            else:
                 logger.error(f"Could not find 'results' in reasoning agent run output: {reasoning_agent_run_result}")
                 raise ValueError("Invalid output format from reasoning agent.")

        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON from reasoning result: {e}")
             logger.error(f"JSON string was: {result_json_string}")
             raise ValueError("Reasoning agent returned invalid JSON.") from e
        except AttributeError as e:
             logger.error(f"Missing expected attribute in reasoning agent result: {e}")
             logger.error(f"Result object was: {reasoning_agent_run_result}")
             raise ValueError("Unexpected output structure from reasoning agent.") from e
        except Exception as e:
            logger.error(f"Error in reasoning step: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        validation_input = {
            "func_name": "validate",
            "problem": module_run.inputs.problem,
            "thoughts": reasoning_output['thoughts']
        }

        validation_run_input = AgentRunInput(
            consumer_id=module_run.consumer_id,
            inputs=validation_input,
            deployment=self.agent_deployments[1], 
            signature=sign_consumer_id(module_run.consumer_id, private_key)
        )

        validation_output = None 
        try:
            validation_agent_run_result = await self.validation_agent.run(validation_run_input)
            logger.info(f"Raw validation agent run result object type: {type(validation_agent_run_result)}")

            if hasattr(validation_agent_run_result, 'results') and validation_agent_run_result.results:
                 result_json_string = validation_agent_run_result.results[0]
                 validation_output = json.loads(result_json_string)
                 best_thought_idx = validation_output.get('best_thought_index', 0) 
                 logger.info(f"Validation completed, selected best thought index: {best_thought_idx}")
            else:
                 logger.error(f"Could not find 'results' in validation agent run output: {validation_agent_run_result}")
                 raise ValueError("Invalid output format from validation agent.")


        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON from validation result: {e}")
             logger.error(f"JSON string was: {result_json_string}")
             raise ValueError("Validation agent returned invalid JSON.") from e
        except AttributeError as e:
             logger.error(f"Missing expected attribute in validation agent result: {e}")
             logger.error(f"Result object was: {validation_agent_run_result}")
             raise ValueError("Unexpected output structure from validation agent.") from e
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e

        result = {
            "problem": module_run.inputs.problem,
            "reasoning_thoughts": reasoning_output['thoughts'],
            "validation_result": validation_output, 
            "final_answer": validation_output.get('final_answer', '') 
        }
        return result

async def run(module_run: dict, *args, **kwargs):
    if not os.getenv("PRIVATE_KEY"):
         logger.error("PRIVATE_KEY environment variable is not set. Cannot sign requests.")
         raise ValueError("Missing PRIVATE_KEY for signing.")

    if 'deployment' not in module_run or not module_run['deployment'].get('name'):
        raise ValueError("Incoming module_run is missing deployment information or deployment name.")

    try:
        orchestrator_run_input = OrchestratorRunInput(**module_run)
        orchestrator_run_input.inputs = InputSchema(**orchestrator_run_input.inputs)
    except Exception as e:
        logger.error(f"Failed to parse incoming module_run data: {e}")
        logger.error(f"Data received: {module_run}")
        raise ValueError("Invalid input data structure for orchestrator run.") from e

    orchestrator = ReasoningValidationOrchestrator()

    try:
        await orchestrator.create(orchestrator_run_input.deployment, *args, **kwargs)
        result = await orchestrator.run(orchestrator_run_input, *args, **kwargs)
        return result
    except Exception as e:
         logger.error(f"Orchestrator execution failed: {e}")
         logger.error(traceback.format_exc())
        
         raise 


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    node_url = os.getenv("NODE_URL")
    private_key_pem = os.getenv("PRIVATE_KEY")

    if not node_url or not private_key_pem:
        raise ValueError("Please set NODE_URL and PRIVATE_KEY environment variables for testing.")

    naptha = Naptha() 

    try:
        deployment = asyncio.run(setup_module_deployment(
            "orchestrator",
            "reasoning_validation_orchestrator/configs/deployment.json", 
            node_url=node_url
        ))

        input_params = {
            "problem": "What is the sum of all integers from 1 to 100?",
            "num_thoughts": 3
        }

        module_run_dict = {
            "inputs": input_params,
            "deployment": deployment.dict(), 
            "consumer_id": naptha.user.id, 
            
        }

        response = asyncio.run(run(module_run_dict)) 

    
        print("\n--- Orchestrator Test Run Complete ---")
        print(f"Problem: {response.get('problem')}")
        print(f"Final Answer: {response.get('final_answer')}")
        best_thought = response.get("validation_result", {}).get("best_thought", "N/A")
        print(f"Best Reasoning Path:\n{best_thought}")
        print("--- Full Response ---")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"\n--- Orchestrator Test Run Failed ---")
        print(f"Error: {e}")
        print(traceback.format_exc())