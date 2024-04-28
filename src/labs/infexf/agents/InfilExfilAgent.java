package src.labs.infexf.agents;

// SYSTEM IMPORTS
import edu.bu.labs.infexf.agents.SpecOpsAgent;
import edu.bu.labs.infexf.distance.DistanceMetric;
import edu.bu.labs.infexf.graph.Vertex;
//import edu.bu.labs.infexf.graph.Path;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
//import edu.cwru.sepia.util.DistanceMetrics;
//import edu.cwru.sepia.util.Pair;


// JAVA PROJECT IMPORTS


public class InfilExfilAgent
    extends SpecOpsAgent
{

    public InfilExfilAgent(int playerNum)
    {
        super(playerNum);
    }

    // if you want to get attack-radius of an enemy, you can do so through the enemy unit's UnitView
    // Every unit is constructed from an xml schema for that unit's type.
    // We can lookup the "range" of the unit using the following line of code (assuming we know the id):
    //     int attackRadius = state.getUnit(enemyUnitID).getTemplateView().getRange();


    @Override
    public float getEdgeWeight(Vertex src,
                               Vertex dst,
                               StateView state)
    {
        
        float baseWeight = 1f;
        float dangerWeight = 100f;
        float weight = baseWeight;
        
        Map<Integer, Vertex> enemyPositions = new HashMap<>();
        Map<Integer, Integer> attackRadii = new HashMap<>();

        for(Integer enemyUnitID : getOtherEnemyUnitIDs()) {
            
            UnitView enemyUnit = state.getUnit(enemyUnitID);
            int attackRadius = enemyUnit.getTemplateView().getRange() + 1;
            Vertex enemyPosition = new Vertex(enemyUnit.getXPosition(), enemyUnit.getYPosition());
            
            enemyPositions.put(enemyUnitID, enemyPosition);
            attackRadii.put(enemyUnitID, attackRadius);

        }

        for (Map.Entry<Integer, Vertex> entry : enemyPositions.entrySet()) {
            Vertex enemy = entry.getValue();
            float dangerDist = DistanceMetric.chebyshevDistance(dst, enemy);
            int attackRadius = attackRadii.get(entry.getKey());
            
            if (dangerDist <= attackRadius) {
                float adjustedWeight = dangerWeight * ((attackRadius / dangerDist));
                weight += adjustedWeight; 
            }
        }
        return weight;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        Stack<Vertex> path = getCurrentPlan();
        
        Map<Integer, Vertex> enemyPositions = new HashMap<>();
        Map<Integer, Integer> attackRadii = new HashMap<>();

        for (Integer enemyID : getOtherEnemyUnitIDs()) {

            UnitView enemyUnit = state.getUnit(enemyID);
            if (enemyUnit == null) {
                continue; // Skip this enemyID if the unit view is null
            }
            int attackRadius = enemyUnit.getTemplateView().getRange() + 1;
            Vertex enemy = new Vertex(enemyUnit.getXPosition(), enemyUnit.getYPosition());
            
            enemyPositions.put(enemyID, enemy);
            attackRadii.put(enemyID, attackRadius);
            
        }

        for (Vertex unit : path) {
            for (Integer enemyID : enemyPositions.keySet()) {

                Vertex enemyPosition = enemyPositions.get(enemyID);
                int attackRadius = attackRadii.get(enemyID);
                float dangerDist = DistanceMetric.chebyshevDistance(unit, enemyPosition);
                if (dangerDist <= attackRadius) {
                    return true;
                }
            }
        }

        return false;
    }

}
