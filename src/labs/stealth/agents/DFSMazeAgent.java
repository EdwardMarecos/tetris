package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


import java.util.Collection;
import java.util.HashSet;   // will need for dfs
import java.util.Stack;     // will need for dfs
import java.util.Set;       // will need for dfs


// JAVA PROJECT IMPORTS


public class DFSMazeAgent
    extends MazeAgent
{

    public DFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }
    public Collection<Vertex> getNeighbors(Vertex v, StateView state) {
        Collection<Vertex> neighbors = new HashSet<>();

        for (int xOffset : new int[]{-1, 0, +1}) {
            for (int yOffset : new int[]{-1, 0, +1}) {
                if (!(xOffset == 0 && yOffset == 0)) {
                    Vertex potentialNeighbor = new Vertex(
                            v.getXCoordinate() + xOffset,
                            v.getYCoordinate() + yOffset
                    );

                    if (state.inBounds(potentialNeighbor.getXCoordinate(), potentialNeighbor.getYCoordinate())
                            && !state.isResourceAt(potentialNeighbor.getXCoordinate(), potentialNeighbor.getYCoordinate())
                            && (!state.isUnitAt(potentialNeighbor.getXCoordinate(), potentialNeighbor.getYCoordinate())
                            || state.unitAt(potentialNeighbor.getXCoordinate(), potentialNeighbor.getYCoordinate()) == this.getEnemyTargetUnitID())) {
                        neighbors.add(potentialNeighbor);
                    }
                }
            }
        }
        return neighbors;
    }

    @Override
    public Path search(Vertex src, Vertex goal, StateView state) {
        Stack<Path> stack = new Stack<>();
        Set<Vertex> visitedVertices = new HashSet<>();

        stack.push(new Path(src));

        while (!stack.isEmpty()) {
            Path currentPath = stack.pop();
            Vertex currentVertex = currentPath.getDestination();

            if (!visitedVertices.contains(currentVertex)) {
                visitedVertices.add(currentVertex);

                for (Vertex neighbor : this.getNeighbors(currentVertex, state)) {
                    if (neighbor.equals(goal)) {
                        return currentPath;
                    }
                    if (!visitedVertices.contains(neighbor)) {
                        stack.push(new Path(neighbor, 1f, currentPath)); // Add new path with the neighbor
                    }
                }
            }
        }

        return null; 
    }

    @Override
    public boolean shouldReplacePlan(StateView state) {
        return false;
    }

}
