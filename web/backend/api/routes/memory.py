"""
Memory API Routes

Endpoints for accessing Atlas's memory systems - this is how the world
can observe what Atlas has learned and remembered.
"""

from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class MemoryItem(BaseModel):
    """A memory item."""
    id: int
    content: Optional[str] = None
    timestamp: Optional[str] = None
    summary: Optional[str] = None
    importance: Optional[float] = None
    name: Optional[str] = None
    category: Optional[str] = None
    connections: Optional[int] = None


class MemoryContents(BaseModel):
    """Memory contents response."""
    memory_type: str
    timestamp: str
    items: List[MemoryItem]
    error: Optional[str] = None


class MemoryQuery(BaseModel):
    """Query for memory search."""
    query: str = Field(..., description="Search query")
    memory_type: str = Field("all", description="Type of memory to search")
    limit: int = Field(10, description="Maximum results to return")


@router.get("/episodic", response_model=MemoryContents)
async def get_episodic_memory(
    request: Request,
    limit: int = 100
):
    """
    Get episodic memories.

    Retrieves specific experiences that Atlas has stored. Episodic memory
    contains time-stamped records of what Atlas has "experienced".

    **Output**: List of episodic memories with timestamps and importance
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_memory_contents(memory_type="episodic", limit=limit)


@router.get("/semantic", response_model=MemoryContents)
async def get_semantic_memory(
    request: Request,
    limit: int = 100
):
    """
    Get semantic memory (concepts and knowledge).

    Retrieves the concept graph that Atlas has built. Semantic memory
    contains abstract knowledge and relationships between concepts.

    **Output**: List of concepts with categories and connection counts
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_memory_contents(memory_type="semantic", limit=limit)


@router.get("/working", response_model=MemoryContents)
async def get_working_memory(request: Request):
    """
    Get current working memory contents.

    Retrieves the active items in Atlas's working memory - what it's
    currently "thinking about" or attending to.

    **Output**: Currently active memory items
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_memory_contents(memory_type="working", limit=100)


@router.get("/associations")
async def get_associations(request: Request):
    """
    Get cross-modal associations.

    Retrieves the associations Atlas has learned between visual and
    auditory patterns - what sounds go with what sights.

    **Output**: Association patterns and strengths
    """
    atlas_manager = request.app.state.atlas_manager

    if atlas_manager.system:
        try:
            associations = atlas_manager.system.analyze_associations()
            return {
                "timestamp": associations.get("timestamp"),
                "total_associations": associations.get("total", 0),
                "top_associations": associations.get("top_associations", []),
                "visual_to_audio": associations.get("visual_to_audio", []),
                "audio_to_visual": associations.get("audio_to_visual", [])
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": None
            }
    else:
        # Demo data
        return {
            "timestamp": None,
            "total_associations": 150,
            "top_associations": [
                {"visual_pattern": "face", "audio_pattern": "speech", "strength": 0.85},
                {"visual_pattern": "motion", "audio_pattern": "rhythm", "strength": 0.72},
                {"visual_pattern": "color_red", "audio_pattern": "alert_tone", "strength": 0.65}
            ]
        }


@router.post("/search")
async def search_memory(request: Request, query: MemoryQuery):
    """
    Search across memory systems.

    Search for specific patterns or concepts in Atlas's memory.

    **Input**: Search query and memory type
    **Output**: Matching memory items
    """
    atlas_manager = request.app.state.atlas_manager

    # In a full implementation, this would use semantic search
    results = {
        "query": query.query,
        "memory_type": query.memory_type,
        "results": [],
        "total_found": 0
    }

    if atlas_manager.system:
        try:
            # Search episodic memory
            if query.memory_type in ["all", "episodic"]:
                episodic = await atlas_manager.get_memory_contents("episodic", query.limit)
                # Filter by query (simple string matching for demo)
                for item in episodic.get("items", []):
                    if query.query.lower() in str(item).lower():
                        results["results"].append({"type": "episodic", **item})

            # Search semantic memory
            if query.memory_type in ["all", "semantic"]:
                semantic = await atlas_manager.get_memory_contents("semantic", query.limit)
                for item in semantic.get("items", []):
                    if query.query.lower() in str(item).lower():
                        results["results"].append({"type": "semantic", **item})

            results["total_found"] = len(results["results"])

        except Exception as e:
            results["error"] = str(e)
    else:
        # Demo results
        results["results"] = [
            {"type": "episodic", "id": 1, "content": f"Demo result for: {query.query}"},
            {"type": "semantic", "id": 2, "content": f"Concept related to: {query.query}"}
        ]
        results["total_found"] = 2

    return results


@router.get("/statistics")
async def get_memory_statistics(request: Request):
    """
    Get memory system statistics.

    Returns overall statistics about Atlas's memory systems.

    **Output**: Memory capacity, usage, and health metrics
    """
    atlas_manager = request.app.state.atlas_manager

    stats = {
        "episodic": {
            "total_memories": 0,
            "capacity_used_percent": 0,
            "oldest_memory": None,
            "newest_memory": None
        },
        "semantic": {
            "total_concepts": 0,
            "total_relationships": 0,
            "average_connections": 0
        },
        "working": {
            "current_items": 0,
            "capacity": 7,  # Miller's magic number
            "attention_focus": None
        }
    }

    if atlas_manager.system:
        try:
            state = atlas_manager.system.get_system_state()

            # Extract memory stats from system state
            if "memory_stats" in state:
                stats.update(state["memory_stats"])

        except Exception as e:
            stats["error"] = str(e)
    else:
        # Demo statistics
        stats["episodic"]["total_memories"] = 1247
        stats["episodic"]["capacity_used_percent"] = 24.5
        stats["semantic"]["total_concepts"] = 89
        stats["semantic"]["total_relationships"] = 342
        stats["semantic"]["average_connections"] = 3.8
        stats["working"]["current_items"] = 4

    return stats
