package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        /*
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int hiddenDim = 2 * numPixelsInImage;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;*/
        final int inputSize = 2 * Board.NUM_COLS + 2;  // input vector size based on board configuration
        final int hiddenDim1 = 2 * inputSize;  // size of hidden layer
        final int hiddenDim2 = inputSize;
        final int outDim = 1;  // output layer size, outputting a single scalar value (q-value)

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputSize, hiddenDim1));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim1, hiddenDim2));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim2, outDim)); // No activation function for the output layer
    
        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Board simulatedBoard = new Board(game.getBoard());
        simulatedBoard.addMino(potentialAction);
        
        int rows = simulatedBoard.NUM_ROWS;
        int cols = simulatedBoard.NUM_COLS;
        
        // List to store column heights. numbers here are height of towers not coords
        // to find their respective row val, do rows - height
        int[] columnHeights = new int[cols];
        // number of holes, badness of a board config
        int holes = 0;  
    
        //logic for determining heights and hole locations
        for (int row = 0; row < rows; row++) {          // iterate through rows
            for (int col = 0; col < cols; col++) {      // iterate through cols
                if (columnHeights[col] == 0) {          // go if we have yet to find a peak
                    if (simulatedBoard.isCoordinateOccupied(col, row)) {
                        columnHeights[col] = rows - row;   // record peak
                    }
                } else {
                    if (!simulatedBoard.isCoordinateOccupied(col, row)) { 
                        // if the spot is unoccupied and has a ceiling, it is a hole
                        holes++;
                    }
                }
            }
        }
        
        // array to record well sizes and locations if applicable
        int[] wells = new int[cols];
        // max height to keep in mind later
        int maxHeight = 0;
        // logic for determining the max peak and well locations (1 <= col <= cols)
        for (int column = 1; column < cols - 1; column++) {
            int prev = columnHeights[column - 1];
            int curr = columnHeights[column];
            int next = columnHeights[column + 1];
            if (curr > maxHeight) {
                maxHeight = curr;
            }
            if (column == 1 && prev > maxHeight) {
                maxHeight = prev;
            }
            if (column == cols - 2 && next > maxHeight) {
                maxHeight = next;
            }
            if(prev > curr && next > curr) {
                int mini = Math.min(prev, next);
                wells[column] = mini - curr;
            }
        }
        // logic for a well at edges
        if (columnHeights[1] > columnHeights[0]){
            wells[0] = columnHeights[0] - columnHeights[1];
        }        
        if (columnHeights[cols - 2] > columnHeights[cols - 1]) {
            wells[cols - 1] = columnHeights[cols - 2] - columnHeights[cols - 1];
        }
        
        /*  effBoard meaning        ~~ logic for constructing effboard as well
        Column 0: num holes
        Column 1: max height
        Columns 2 to cols + 2: column heights
        Columns cols + 3 to 2 cols - 1: well depths
         */
        int features = cols * 2 + 2;
        Matrix effBoard = Matrix.zeros(1, features);
        for (int feature = 0; feature < features; ++feature) {
            // holes
            if (feature == 0) {
                effBoard.set(0, cols, holes); // Set total holes
                break;
            }
            // max
            if (feature == 1) {
                effBoard.set(0, 2 * cols + 1, maxHeight); // Set max height
                break;
            }
            // peaks
            if (feature < cols + 2) {
                effBoard.set(0, feature, columnHeights[feature - 2]);
                break;
            }
            // wells
            effBoard.set(0, feature, wells[feature - cols - 2]);            
        }

        // System.out.println(effBoard.toString());
        // try {
        //     // Pause the thread for 5000 milliseconds (5 seconds)
        //     Thread.sleep(4000);
        // } catch (InterruptedException e) {
        //     // Handle the interruption exception
        //     System.err.println("The sleep operation was interrupted.");
        // }
        return effBoard;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        /*return this.getRandom().nextDouble() <= EXPLORATION_PROB;*/
        // Factors to adjust exploration based on game progress and total training progress
        double initialExplorationRate = 0.05;
        double moveDecayRate = 0.1; // Higher value encourages less exploration as the game progresses
        double gameDecayRate = 5000; // Slower decay rate across games

        // Calculate exploration probability that decreases as more moves are made and more games are played
        double moveBasedDecay = Math.exp(-gameCounter.getCurrentMoveIdx() * moveDecayRate);
        double gameBasedDecay = Math.exp(-gameCounter.getTotalGamesPlayed() / gameDecayRate);

        double explorationProb = initialExplorationRate * moveBasedDecay * gameBasedDecay;

        return this.getRandom().nextDouble() < explorationProb;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        double reward = 0;
        // reward obtained in current turn
        reward += game.getScoreThisTurn();
        // Get the board representation for further analysis.
        // ------------------------------------------------------------
        int rows = game.getBoard().NUM_ROWS;
        int cols = game.getBoard().NUM_COLS;
        // array to record well sizes and locations if applicable
        int[] wells = new int[cols];
         // List to store column heights. numbers here are height of towers not coords
        // to find their respective row val, do rows - height
        int[] columnHeights = new int[cols];
        // number of holes, badness of a board config
        int holes = 0;          
        // max height to keep in mind later
        int maxHeight = 0;

        //logic for determining heights and hole locations
        for (int row = 0; row < rows; row++) {          // iterate through rows
            for (int col = 0; col < cols; col++) {      // iterate through cols
                if (columnHeights[col] == 0) {          // go if we have yet to find a peak
                    if (game.getBoard().isCoordinateOccupied(col, row)) {
                        columnHeights[col] = rows - row;   // record peak
                    }
                } else {
                    if (!game.getBoard().isCoordinateOccupied(col, row)) { 
                        // if the spot is unoccupied and has a ceiling, it is a hole
                        holes++;
                    }
                }
            }
        }
        // logic for determining the max peak and well locations (1 <= col <= cols)
        for (int column = 1; column < cols - 1; column++) {
            int prev = columnHeights[column - 1];
            int curr = columnHeights[column];
            int next = columnHeights[column + 1];
            if (curr > maxHeight) {
                maxHeight = curr;
            }
            if (column == 1 && prev > maxHeight) {
                maxHeight = prev;
            }
            if (column == cols - 2 && next > maxHeight) {
                maxHeight = next;
            }
            if(prev > curr && next > curr) {
                int mini = Math.min(prev, next);
                wells[column] = mini - curr;
            }
        }
        // logic for a well at edges
        if (columnHeights[1] > columnHeights[0]){
            wells[0] = columnHeights[0] - columnHeights[1];
        }        
        if (columnHeights[cols - 2] > columnHeights[cols - 1]) {
            wells[cols - 1] = columnHeights[cols - 2] - columnHeights[cols - 1];
        }
        // ------------------------------------------------------------

        // REWARDS based on current board alongside 
        // future pieces and how well the board is prepared for them

        // Analyzing preparation for upcoming minos
        List<Mino.MinoType> nextMinos = game.getNextThreeMinoTypes();
        double preparednessReward = 0;
        boolean fail = false; // fail is determined is a height is higher than we can have

        // begin calculations for individual preparations
        int levelStreak = 1; // Count of adjacent columns at the same height
        int lastHeight = -1; // Initialize to an impossible value
        double HEIGHT_PENALTY = 0.1;
        double HOLE_PENALTY = 1.5;
        double MAX_HEIGHT_PENALTY = 2.0;
        double WELL_DEPTH_PENALTY = 0.5;
        double GAME_OVER_PENALTY = 100.0;
        double Ipositions = 0;
        double Jpositions = 0;
        double Lpositions = 0;
        double Opositions = 0;
        double Spositions = 0;
        double Tpositions = 0;
        double Zpositions = 0;
        int tetris = 0; // tetris 4 line clear at base of graph
        int tspin = 0; // since i dont track holes a well of 3 or more is enough for me to consider t-spins
        for (int height : columnHeights) {
            if (height == rows) {fail = true;}
            if (height == lastHeight) { // Check if the height matches the first column's height
                levelStreak++;
            } else {
                levelStreak = 1; // Reset streak if heights don't match
                lastHeight = height;
            }
            Ipositions += .5; // 'I' Mino standing less than optimal
            Jpositions += .5; // same
            Tpositions += .5; // t on side
            Spositions += .5; // s or z standing
            Zpositions += .5;
            if (levelStreak >=2 ) { // found a potential space for 'O' Mino
                Opositions++;
                Spositions++;
                Zpositions++;
                Lpositions += .5; // l standing (still good overall)
            }
            if (levelStreak >= 3) {
                Lpositions++;
                Jpositions++;
                Tpositions++; // 'T' lying flat
            }
            if (levelStreak >= 4) { // Found a place for 'I' Mino
                Ipositions++;
            }
        }
        for (int wellDepth =0; wellDepth < wells.length; wellDepth++) {
            if (wells[wellDepth] == 1) {
                Lpositions += .5; // useful if they go in sideways
                Jpositions += .5;
                Tpositions += .5; // t s or z
                Spositions += .5;
                Zpositions += .5;
            }
            if (wells[wellDepth] == 2) {
                Lpositions++;
                Jpositions++;
            }
            if (wells[wellDepth] >= 3) { // Count wells at least 3 blocks deep as useful
                Ipositions++;
                tspin++;
            }
            if (wells[wellDepth] >= 4 && columnHeights[wellDepth] == 0) { 
                //a very positive position sets us up for tetris line clear
                tetris++;
                tspin++;
            }
        }
        // finish calculation for individual preparations
        
        for (Mino.MinoType upcoming : nextMinos) {
            if (upcoming == Mino.MinoType.I) {
                preparednessReward += 2 * Ipositions + 50 * tetris;
                break; 
            }
            if (upcoming == Mino.MinoType.J) {
                preparednessReward += 2 * Jpositions;
                break;
            }
            if (upcoming == Mino.MinoType.L) {
                preparednessReward += 2 * Lpositions;
                break;
            }
            if (upcoming == Mino.MinoType.O) {
                preparednessReward += 2 * Opositions;
                break;
            }
            if (upcoming == Mino.MinoType.S) {
                preparednessReward += 2 * Spositions;
                break;
            }
            if (upcoming == Mino.MinoType.T) {
                preparednessReward += 2 * Tpositions + tspin;
                break;
            }
            if (upcoming == Mino.MinoType.Z) {
                preparednessReward += 2 * Zpositions;
                break;
            }
        }
        
        reward += preparednessReward;

        // punishments based on current board, overall unfavorable
        
        // minor penalty for each column's height
    
        for (int height : columnHeights) {
            reward -= Math.pow(height, 2) * HEIGHT_PENALTY;  // Exponential penalty for column height
        }
    
        reward -= (holes > 0) ? Math.pow(holes, 2) * HOLE_PENALTY : 0;  // Exponential penalty for holes
    
        for (int depth : wells) {
            reward -= Math.pow(depth, 2) * WELL_DEPTH_PENALTY;  // Exponential penalty for well depth
        }
        
        reward -= maxHeight * MAX_HEIGHT_PENALTY;

        // Heavy penalty for losing the game
        if (fail) {
            reward -= GAME_OVER_PENALTY;  // Large penalty for game over
        }
 
        return reward;
    }

}
    