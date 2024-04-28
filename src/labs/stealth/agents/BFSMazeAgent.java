package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;

import java.util.Collection;
import java.util.HashSet;       // will need for bfs
import java.util.Queue;         // will need for bfs
import java.util.LinkedList;    // will need for bfs
import java.util.Set;           // will need for bfs


// JAVA PROJECT IMPORTS


public class BFSMazeAgent
    extends MazeAgent
{

    public BFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }
    public Collection<Vertex> getNeighbors(Vertex v, StateView state) {
        Collection<Vertex> neighbors = new HashSet<Vertex>();

        for(int xOffset : new int[]{-1, 0, +1}) {
            for(int yOffset : new int[]{-1, 0, +1}) {
                if(!(xOffset == 0 && yOffset == 0)) {
                    Vertex potentialNeighbor = new Vertex(
                        v.getXCoordinate() + xOffset,
                        v.getYCoordinate() + yOffset
                    );

                    if(state.inBounds(potentialNeighbor.getXCoordinate(), potentialNeighbor.getYCoordinate())
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
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Queue<Path> queue = new LinkedList<Path>();
        Set<Vertex> visitedVertices = new HashSet<Vertex>();

        // populate queue with initial src
        queue.add(new Path(src));
        visitedVertices.add(src);

        while(!queue.isEmpty())
        {
            // grab the first Path
            Path currentPath = queue.poll();

            // add all unvisited neighbors
            for(Vertex neighbor : this.getNeighbors(currentPath.getDestination(), state))
            {
                if(neighbor.equals(goal))
                {
                    return currentPath;
                }
                if(!visitedVertices.contains(neighbor))
                {
                    queue.add(new Path(neighbor, 1f, currentPath));
                    visitedVertices.add(neighbor);
                }
            }
        }

        return null;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        return false;
    }

}
