def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    new_values = []

    for s in range(len(values)):
        best_value = float("-inf")

        for a in range(len(transitions[s])):
            expected_value = 0.0

            for next_state, prob in enumerate(transitions[s][a]):
                expected_value += prob * values[next_state]

            action_value = rewards[s][a] + gamma * expected_value
            best_value = max(best_value, action_value)

        new_values.append(best_value)

    return new_values