from madqn import MADQN
import pickle

if __name__ == '__main__':
    # train and save a network
    algorithm = MADQN(mode='train')
    algorithm.train(num_episodes=110)

    # test a network or the heuristic
    # filename = None
    # algorithm = MADQN(mode='test', filename=filename)
    # algorithm.test(num_episodes=1, method='network')

    # save the results to file
    # results_filename = 'results.pkl'
    # output = open(filename, 'wb')
    # pickle.dump(results, output)
    # output.close()
