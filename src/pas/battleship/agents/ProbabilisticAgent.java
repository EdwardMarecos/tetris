package src.pas.battleship.agents;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.battleship.agents.Agent;
import edu.bu.battleship.game.Game.GameView;
import edu.bu.battleship.game.ships.Ship.ShipType;
import edu.bu.battleship.game.Constants;
import edu.bu.battleship.game.EnemyBoard;
import edu.bu.battleship.game.EnemyBoard.Outcome;
import edu.bu.battleship.utils.Coordinate;


public class ProbabilisticAgent
    extends Agent
{

    public ProbabilisticAgent(String name)
    {
        super(name);
        System.out.println("[INFO] ProbabilisticAgent.ProbabilisticAgent: constructed agent");
    }

    // cache to store coordinates adjacent to hits
    private Set<Coordinate> adjacentHitsCache = new HashSet<>();

    @Override
    public Coordinate makeMove(final GameView game)
    {
        
        Set<Coordinate> extensionPoints = findExtensionsOfHits(game);
        Coordinate nextMove = selectNextMove(extensionPoints, game);
        if (nextMove != null) 
        {
            return nextMove;
        }

        // initialize a probability grid
        int rows = game.getGameConstants().getNumRows();
        int cols = game.getGameConstants().getNumCols();
        double[][] probabilityGrid = new double[rows][cols];


        // Early selection for squares directly adjacent to hits
        for (Coordinate adjacentHit : adjacentHitsCache) 
        {
            EnemyBoard.Outcome outcome = game.getEnemyBoardView()[adjacentHit.getXCoordinate()][adjacentHit.getYCoordinate()];
            if (outcome == EnemyBoard.Outcome.UNKNOWN) 
            { // If the square hasn't been tried yet
                return adjacentHit; // Select this square immediately
            }
        }
        // get enemy board information
        EnemyBoard.Outcome[][] enemyBoardView = game.getEnemyBoardView();
        // how many of each unit remain
        Map<ShipType, Integer> enemyShipTypeToNumRemaining = game.getEnemyShipTypeToNumRemaining();
    
        //  iterate through enemy board to know 
        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                // we already know the outcome at this spot
                if (enemyBoardView[r][c] != EnemyBoard.Outcome.UNKNOWN)
                {
                    continue;
                }
                for (ShipType shipType : ShipType.values())
                {
                    // get the ship size for each ship type
                    int shipSize = Constants.Ship.getShipSize(shipType);
                    // 
                    int remainingShipsOfType = enemyShipTypeToNumRemaining.getOrDefault(shipType, 0);
                    
                    // use helper function to find probability of each cell
                    double probability = calcProbability(r, c, shipType, shipSize, remainingShipsOfType, game, probabilityGrid);
                    probabilityGrid[r][c] += probability;
                }
            }
        }

        // determine coord with best move
        Coordinate bestMove = selectBestMove(probabilityGrid, game);
        return bestMove;
    }

    // calculate the probability of a ship being placed in a given location
    private double calcProbability(int row, int col, ShipType ship, int size, int remaining, GameView game, double[][] pGrid)
    {
        // init prob bool
        double prob = 0.0;
        
        // if there are no more availiable ships of *type*, impossible for that ship to be on the unknown
        if (remaining == 0)
        {
            return prob;
        }
        
        // check for horizontal ships
        for (int startCol = col - size + 1; startCol <= col; startCol++)
        {
            if (canPlaceShip(row, startCol, size, true, game))
            {
                prob += calcProbPlacement(row, startCol, size, true, game);
            }
        }

        // check for vertical ships
        for (int startRow = row - size + 1; startRow <= row; startRow++)
        {
            if (canPlaceShip(startRow, col, size, false, game))
            {
                prob += calcProbPlacement(startRow, col, size, false, game);
            }
        }

        return prob;
    }

    private boolean canPlaceShip(int StartRow, int StartCol, int size, boolean horizontal, GameView game)
    {
        EnemyBoard.Outcome[][] enemyBoardView = game.getEnemyBoardView();
        int rows = game.getGameConstants().getNumRows();
        int cols = game.getGameConstants().getNumCols();

        // If placing horizontally, iterate over columns; otherwise, iterate over rows.
        for (int i = 0; i < size; i++) 
        {
            int currentRow = StartRow + (horizontal ? 0 : i);
            int currentCol = StartCol + (horizontal ? i : 0);

            // Check bounds.
            if (currentRow < 0 || currentRow >= rows || currentCol < 0 || currentCol >= cols) 
            {
                return false; // Part of the ship would be out of bounds.
            }

            // Check if the cell is already known to be a miss or part of a sunk ship.
            if (enemyBoardView[currentRow][currentCol] == EnemyBoard.Outcome.MISS ||
                enemyBoardView[currentRow][currentCol] == EnemyBoard.Outcome.SUNK) 
            {
                return false; // Can't place a ship on a cell that is a known miss or part of a sunk ship.
            }
        }

        return true; // The ship can be placed.
    }

    private double calcProbPlacement(int startRow, int startCol, int size, boolean horizontal, GameView game) {
        EnemyBoard.Outcome[][] enemyBoardView = game.getEnemyBoardView();
        
        double baseProbability = 1.0; // Base probability without considering adjacency to hits
        int validPositions = 0; // Count of valid positions for the ship
    
        for (int i = 0; i < size; i++) {
            int currentRow = horizontal ? startRow : startRow + i;
            int currentCol = horizontal ? startCol + i : startCol;
    
            // Check if the current position is within bounds and not a known miss or sunk
            if (currentRow < 0 || currentRow >= enemyBoardView.length || 
                currentCol < 0 || currentCol >= enemyBoardView[0].length ||
                enemyBoardView[currentRow][currentCol] == EnemyBoard.Outcome.MISS ||
                enemyBoardView[currentRow][currentCol] == EnemyBoard.Outcome.SUNK) {
                return 0.0; // Invalidate this placement if out of bounds or on a known miss/sunk
            }
    
            // Increase the count for each valid position
            if (enemyBoardView[currentRow][currentCol] == EnemyBoard.Outcome.UNKNOWN) {
                validPositions++;
            }
        }
    
        // Adjust the probability based on the number of valid positions
        if (validPositions > 0) {
            baseProbability = (double) validPositions / size;
        } else {
            return 0.0; // No valid positions for this ship orientation and size
        }
    
        return baseProbability;
    }
    
    private Coordinate selectBestMove(double[][] probabilityGrid, GameView game) 
    {
    List<Coordinate> candidates = new ArrayList<>();
    double highestProbability = -1.0;

    // First pass: find the highest probability
    for (int r = 0; r < probabilityGrid.length; r++) {
        for (int c = 0; c < probabilityGrid[r].length; c++) 
        {
            if (probabilityGrid[r][c] > highestProbability) 
            {
                highestProbability = probabilityGrid[r][c];
                candidates.clear(); // Clear candidates as we found a higher probability
                candidates.add(new Coordinate(r, c));
            } else if (probabilityGrid[r][c] == highestProbability) 
            {
                // Add coordinates with equal highest probability to candidates
                candidates.add(new Coordinate(r, c));
            }
        }
    }

    // Default to the first candidate if  only one candidate exists
    return candidates.get(0);
}
    
    // prioritize spaces adjacent to known hits
    // find possible extension points
    private Set<Coordinate> findExtensionsOfHits(GameView game) 
    {
        Set<Coordinate> potentialExtensions = new HashSet<>();
        EnemyBoard.Outcome[][] board = game.getEnemyBoardView();

        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[r].length; c++) 
            {
                // Skip if not a hit or already processed in a sequence
                if (board[r][c] != EnemyBoard.Outcome.HIT || alreadyProcessedInSequence(r, c, potentialExtensions)) 
                {
                    continue;
                }

                // Check for horizontal and vertical sequences from this hit
                boolean horizontalSequence = hasHit(board, r, c + 1) || hasHit(board, r, c - 1);
                boolean verticalSequence = hasHit(board, r + 1, c) || hasHit(board, r - 1, c);

                if (horizontalSequence) 
                {
                    addExtensionPoints(potentialExtensions, board, r, c, true); // True for horizontal
                } 
                if (verticalSequence) 
                {
                    addExtensionPoints(potentialExtensions, board, r, c, false); // False for vertical
                }
                if (!horizontalSequence && !verticalSequence) 
                {
                    // It's an isolated hit, add all unknown adjacent tiles
                    addAllCardinalDirections(potentialExtensions, board, r, c);
                }
            }
        }

        return potentialExtensions;
    }

    private boolean alreadyProcessedInSequence(int row, int col, Set<Coordinate> processedHits) 
    {
        // Check if the given coordinate is close to any in the processedHits set.
        for (Coordinate processedHit : processedHits) 
        {
            if (Math.abs(processedHit.getXCoordinate() - row) <= 1 && Math.abs(processedHit.getYCoordinate() - col) <= 1) 
            {
                return true;
            }
        }
        return false;
    }

    private boolean hasHit(EnemyBoard.Outcome[][] board, int row, int col) 
    {
        // Check if the coordinates are within the board boundaries
        if (row >= 0 && row < board.length && col >= 0 && col < board[row].length) 
        {
            // Return true if the cell at the given coordinates is a hit
            return board[row][col] == EnemyBoard.Outcome.HIT;
        }
        // Return false if the coordinates are out of bounds
        return false;
    }
    
    private void addExtensionPoints(Set<Coordinate> potentialExtensions, EnemyBoard.Outcome[][] board, int row, int col, boolean isHorizontal) 
    {
        int[] direction = isHorizontal ? new int[]{0, 1} : new int[]{1, 0}; // Right for horizontal, Down for vertical
        int oppositeDirectionRow = isHorizontal ? 0 : -1; // Up for vertical
        int oppositeDirectionCol = isHorizontal ? -1 : 0; // Left for horizontal
    
        // Find the forward extension point
        int forwardRow = row, forwardCol = col;
        while (isValid(board, forwardRow + direction[0], forwardCol + direction[1]) && 
               board[forwardRow + direction[0]][forwardCol + direction[1]] == EnemyBoard.Outcome.HIT) 
        {
            forwardRow += direction[0];
            forwardCol += direction[1];
        }
    
        // Check and add the forward unknown tile if it exists
        if (isValid(board, forwardRow + direction[0], forwardCol + direction[1]) && 
            board[forwardRow + direction[0]][forwardCol + direction[1]] == EnemyBoard.Outcome.UNKNOWN) 
        {
            potentialExtensions.add(new Coordinate(forwardRow + direction[0], forwardCol + direction[1]));
        }
    
        // Find the backward extension point
        int backwardRow = row, backwardCol = col;
        while (isValid(board, backwardRow + oppositeDirectionRow, backwardCol + oppositeDirectionCol) &&
               board[backwardRow + oppositeDirectionRow][backwardCol + oppositeDirectionCol] == EnemyBoard.Outcome.HIT) 
        {
            backwardRow += oppositeDirectionRow;
            backwardCol += oppositeDirectionCol;
        }
    
        // Check and add the backward unknown tile if it exists
        if (isValid(board, backwardRow + oppositeDirectionRow, backwardCol + oppositeDirectionCol) && 
            board[backwardRow + oppositeDirectionRow][backwardCol + oppositeDirectionCol] == EnemyBoard.Outcome.UNKNOWN) 
        {
            potentialExtensions.add(new Coordinate(backwardRow + oppositeDirectionRow, backwardCol + oppositeDirectionCol));
        }
    }
    
    private void addAllCardinalDirections(Set<Coordinate> potentialExtensions, EnemyBoard.Outcome[][] board, int row, int col) 
    {
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // Up, Down, Left, Right
    
        for (int[] dir : directions) 
        {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
    
            // Check if the new coordinate is within the board and unknown
            if (isValid(board, newRow, newCol) && board[newRow][newCol] == EnemyBoard.Outcome.UNKNOWN) 
            {
                potentialExtensions.add(new Coordinate(newRow, newCol));
            }
        }
    }
    
    private boolean isValid(EnemyBoard.Outcome[][] board, int row, int col) 
    {
        // Check if the given row and col are within the bounds of the board
        return row >= 0 && row < board.length && col >= 0 && col < board[0].length;
    } 

    // select an extension from list
    private Coordinate selectNextMove(Set<Coordinate> extensionPoints, GameView game) 
    {
           
        // Convert to list for easy indexing
        List<Coordinate> extensionList = new ArrayList<>(extensionPoints);
        
        if (!extensionList.isEmpty()) 
        {
            Random random = new Random();
            return extensionList.get(random.nextInt(extensionList.size()));
        }
    
        return null; 
    }
    
    @Override
    public void afterGameEnds(final GameView game) {}

}
