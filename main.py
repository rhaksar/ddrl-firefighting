from madqn import MADQN
import pickle

if __name__ == '__main__':
    # train and save a network
    algorithm = MADQN(mode='train')
    algorithm.train(num_episodes=110)

    # test a network or the heuristic
    # load_filename = 
    # test_method = 'network'
    # algorithm = MADQN(mode='test', filename=load_filename)
    # results = algorithm.test(num_episodes=1, method=test_method)

    # save the results to file
    # WARNING: the resulting output file may be very large, ~1.5 GB
    # save_filename = 'results_' + test_method + '.pkl'
    # output = open(save_filename, 'wb')
    # pickle.dump(results, output)
    # output.close()
