import sys
import os
from pathlib import Path

# Add project root and framework to path
project_root = Path(__file__).resolve().parent
framework_path = project_root.parent / "rag-pipeline-framework"
sys.path.append(str(project_root))
sys.path.append(str(framework_path))

from data_ingestion.docling_loader import DoclingPDFLoader
from data_ingestion.prepare_chunks import IPLChunkPreparer
from ipl_retrieval.vector_store import IPLVectorStore
from graph.workflow import app

# Evaluation Set
EVAL_QUERIES = [
    {"id": "E1", "query": "What is the minimum CGPA to join RCB's squad?"},
    {"id": "E2", "query": "Who captains Chennai Super Kings in 2024?"},
    {"id": "E3", "query": "How many IPL titles has Mumbai Indians won?"},
    {"id": "E4", "query": "What is Virat Kohli's career IPL run tally?"},
    {"id": "E5", "query": "Who has taken the most wickets in IPL history?"},
    {"id": "E6", "query": "What is the highest team total in IPL history?"},
    {"id": "E7", "query": "What type of pitch is the Chinnaswamy Stadium known for?"},
    {"id": "E8", "query": "How many matches has MS Dhoni played in IPL?"},
    {"id": "M1", "query": "Which teams won the IPL title more than once between 2019–2024?"},
    {"id": "M2", "query": "List all bowlers with an economy rate below 7.0."},
    {"id": "M3", "query": "Which opener has the highest strike rate among batters in the dataset?"},
    {"id": "M4", "query": "Compare Jasprit Bumrah and Rashid Khan on all bowling metrics."},
    {"id": "M5", "query": "Which venue has the highest average first-innings score?"},
    {"id": "M6", "query": "How many times have MI and CSK played each other?"},
    {"id": "M7", "query": "Who won the last 5 matches between KKR and RCB?"},
    {"id": "M8", "query": "What is Virat Kohli's form in the last 5 matches?"},
    {"id": "M9", "query": "Which team should bat first at Eden Gardens and why?"},
    {"id": "M10", "query": "Which player has the most centuries in IPL history?"},
    {"id": "H1", "query": "Suggest a Dream11 XI for KKR vs RCB at Eden Gardens."},
    {"id": "H2", "query": "Who is likely to win if RR plays SRH at Wankhede? Justify."},
    {"id": "H3", "query": "Which team has been the most consistent across all seasons from 2019–2024?"},
    {"id": "H4", "query": "What bowling strategy should MI use against RCB at Chinnaswamy?"},
    {"id": "H5", "query": "Compare Rohit Sharma and KL Rahul as IPL captains — batting + team results."},
    {"id": "H6", "query": "Has any conflict been detected in Virat Kohli's career runs data?"},
    {"id": "H7", "query": "Which is better — chasing or defending at Hyderabad? Use historical data."},
    {"id": "X1", "query": "Who should I pick as Dream11 captain for every match this week?"},
    {"id": "X2", "query": "What is Kohli's average against left-arm pace specifically?"},
    {"id": "X3", "query": "Is Yuzvendra Chahal's wicket count 205 or 187?"},
    {"id": "X4", "query": "Predict the IPL 2025 champion."},
    {"id": "X5", "query": "Which player has the highest salary in IPL 2024 auction?"},
    {"id": "X6", "query": "What is the BCCI's net worth?"},
    {"id": "X7", "query": "Tell me everything about cricket."},
    {"id": "X8", "query": "Is Sachin Tendulkar in this IPL dataset?"},
    {"id": "X9", "query": "Who won the Best Batsman award in IPL 2024?"},
    {"id": "X10", "query": "Suggest a team for BCCI's next T20 World Cup squad."}
]

def main():
    print("Initializing IPL RAG Assistant Evaluation Suite...")
    
    # 1. Load Dataset
    loader = DoclingPDFLoader(dataset_path="dataset")
    documents = loader.load_documents()
    
    # 2. Chunk Chunks
    chunk_preparer = IPLChunkPreparer()
    chunks = chunk_preparer.prepare_chunks(documents)
    
    # 3. Populate Vector Store
    vector_store = IPLVectorStore()
    vector_store.store_documents(chunks)
    
    # 4. Open evaluation report file
    report_path = project_root / "evaluation_scorecard.md"
    print(f"Executing 35 Evaluation Queries. Saving report to: {report_path}")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# IPL LangGraph RAG Agent - Official 35-Query Evaluation Scorecard\n\n")
        f.write("| ID | Query | Router Node Type | Entities Extracted | Response Snippet |\n")
        f.write("|---|---|---|---|---|\n")
        
        for item in EVAL_QUERIES:
            qid = item["id"]
            query = item["query"]
            print(f"\n[{qid}] Query: {query}")
            
            try:
                result = app.invoke({
                    "query": query,
                    "documents": documents,
                    "chunks": chunks,
                    "chat_history": [],
                    "last_context": "",
                    "previous_docs": [],
                    "resolved_entities": [],
                    "resolved_entity_type": None
                })
                
                answer = result.get("answer", "No answer found.").strip()
                qtype = result.get("query_type", "N/A")
                entities = result.get("entities", [])
                
                # Format snippet
                snippet = answer.replace("\n", " ").replace("|", "\\|")
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                    
                f.write(f"| {qid} | {query} | {qtype} | {entities} | {snippet} |\n")
                f.flush()
                
                print(f"Type: {qtype} | Entities: {entities}")
                print(f"Answer: {answer[:100]}...")
            except Exception as e:
                f.write(f"| {qid} | {query} | ERROR | N/A | Error running query: {e} |\n")
                print(f"Error: {e}")
                
    print("\nEvaluation completed successfully! Check evaluation_scorecard.md for results.")

if __name__ == "__main__":
    main()
