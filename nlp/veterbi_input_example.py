import numpy as np

input_lattice = ['Noun', 'Verb', 'Adj']
input_word = ['coffee', 'takes', 'free']

input_transition_probability = [[0.4, 0.5, 0.1],
                                [0.4, 0.1, 0.2],
                                [0.5, 0.2, 0.3]]
input_emission_probability = [[0.3, 0.3, 0.4],
                              [0.2, 0.4, 0.4],
                              [0.5, 0.1, 0.4]]


def solution(input_sentence):
    input_sentence_length = len(input_sentence)
    input_lattice_length = len(input_lattice)

    dp = np.zeros((input_lattice_length, input_sentence_length))
    max_path = [-1 for i in range(input_sentence_length)]

    for i in range(input_lattice_length):
        # initiate probability
        dp[i][0] = input_emission_probability[i][0] * 1/3
    # iterate every words
    for j in range(1, input_sentence_length):
        # iterate every lattice
        max_probability_current = -1
        for i in range(input_lattice_length):
            # iterate every lattice to select the max probability of the previous words
            for k in range(input_lattice_length):
                temp_probability = (dp[k][j - 1]) * (input_transition_probability[k][i]) * (
                    input_emission_probability[i][j])
                if dp[i][j] < temp_probability:
                    dp[i][j] = temp_probability

    # find max probability and max path
    max_probability = -1
    for j in range(input_sentence_length):
        max_probability_column = -1
        for i in range(input_lattice_length):
            if j == (input_sentence_length - 1):
                if max_probability < dp[i][input_sentence_length - 1]:
                    max_probability = dp[i][input_sentence_length - 1]
                    max_path[j] = i
            else:
                if max_probability_column < dp[i][j]:
                    max_probability_column = dp[i][j]
                    max_path[j] = i
    return max_path, max_probability


# test
sentence = ['coffee', 'takes', 'free']

print(solution(sentence))
