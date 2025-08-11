import os
import logging
from dotenv import load_dotenv
from agent.graph import build_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    logger.info("Starting AI Study Agent (Plan–Execute)")

    graph = build_graph()

    print("AI Study Agent (Plan–Execute). Type your question (Ctrl+C to quit).")
    while True:
        try:
            q = input("\n> ")
        except KeyboardInterrupt:
            logger.info("Exiting...")
            break
        state = {"question": q, "scratchpad": [], "evidence": [], "plan": None, "answer": None}
        logger.info(f"Received question: {q}")
        final_state = graph.invoke(state)
        logger.info("Agent finished processing.")
        print("\n=== Answer ===")
        print(final_state["answer"])

if __name__ == "__main__":
    main()
