Just knowing how to search is not enough, the Pacman needs survival skills while eating all the dots. In this project, various Pacman agents and agent component are been implemented.

1. ReflexAgent
   - An evaluation function that encourage avoidance to ghost while being closer to foods.
   -  `successorGameState.getScore() + minGhostDist / minFoodDist + score`
              

2. MinimaxAgent
   - A Pacman Agent that utilizes Min-Max Tree Search as its core algorithm. The implementation is very generalized so it can deal with any number of Ghost and any depth.
   ``` python
    def value(self, agentIndex, game_state: GameState, depth):
        if len(game_state.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return self.evaluationFunction(game_state), ""

        if agentIndex == 0:
            return self.maxValue(agentIndex, game_state, depth)
        return self.minValue(agentIndex, game_state, depth)

    def minValue(self, agentIndex, game_state: GameState, depth):
        min_value = float("inf")
        min_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth)[0]
            if cur_value < min_value:
                min_value = cur_value
                min_action = action
        return min_value, min_action

    def maxValue(self, agentIndex, game_state: GameState, depth):
        max_value = float("-inf")
        max_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth)[0]
            if cur_value > max_value:
                max_value = cur_value
                max_action = action
        return max_value, max_action
    ```
3. AlphaBetaAgent
   - A Pacman Agent that is very similar to MinimaxAgent and also uses Min-Max Tree Search but prunes unnecessary tree branches to speed up decision making speed.
   ```python
    def value(self, agentIndex, game_state: GameState, depth, alpha, beta):
        if len(game_state.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return self.evaluationFunction(game_state), ""

        if agentIndex == 0:
            return self.maxValue(agentIndex, game_state, depth, alpha, beta)
        return self.minValue(agentIndex, game_state, depth, alpha, beta)

    def minValue(self, agentIndex, game_state: GameState, depth, alpha, beta):
        min_value = float("inf")
        min_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth, alpha, beta)[0]
            if cur_value < alpha:
                return cur_value, action

            if cur_value < min_value:
                min_value = cur_value
                min_action = action

            beta = min(beta, cur_value)

        return min_value, min_action

    def maxValue(self, agentIndex, game_state: GameState, depth, alpha, beta):
        """
            a min will choose among the max of each node
            if we found a value that is greater than previous min
            this value is the lowest val we will choose, but the min chooser will not choose it
            so no need to calculate later nodes
        """
        max_value = float("-inf")
        max_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth, alpha, beta)[0]
            if cur_value > beta:
                return cur_value, action

            if cur_value > max_value:
                max_value = cur_value
                max_action = action

            alpha = max(alpha, cur_value)

        return max_value, max_action
   ```

4. ExpectimaxAgent
   - Also its very similar to MinimaxAgent but works better in situations that opponents do not make optimal decision.
   ```python
    def expectValue(self, agentIndex, game_state: GameState, depth):
        legal_moves = game_state.getLegalActions(agentIndex)
        expect_value = 0

        for action in legal_moves:
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth)[0]
            expect_value += cur_value

        return expect_value / len(legal_moves), ""
   ```

5. betterEvaluationFunction
   - A evaluation function that evaluate the state of the game.
   ```python
    def betterEvaluationFunction(currentGameState: GameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
    
        DESCRIPTION: <write something here so we know what you did>
        """
        foods = currentGameState.getFood().asList()
        pos = currentGameState.getPacmanPosition()
    
        food_count = len(foods) if foods else 1
        foodDist = [manhattanDistance(pos, food) for food in foods]
        closestFoodDist = min(foodDist) if foodDist else 0.1
    
        return currentGameState.getScore() + 1.0 / closestFoodDist - food_count
   ```