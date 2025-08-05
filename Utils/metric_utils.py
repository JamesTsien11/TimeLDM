## Necessary Packages
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
   print('Final Score: ', f'{mean} \xB1 {sigma}')

def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def visualization(ori_data, generated_data, analysis, compare=3000):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="TimeLDM")

        # ax.legend()
        plt.legend(prop={'size': 15})
        # plt.title('PCA plot')
        # plt.xlabel('x-pca')
        # plt.ylabel('y_pca')
        plt.xticks([-1,0,1,2])
        plt.yticks([-0.5,0.0,0.5])
        plt.tick_params(labelsize=15)
        plt.savefig('data2.png', dpi=1000,bbox_inches='tight')
        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="TimeLDM")

        # ax.legend()
        plt.legend(prop={'size': 15})
        # plt.title('t-SNE plot')
        # plt.xlabel('x-tsne')
        # plt.ylabel('y_tsne,2')
        
        plt.yticks([-10,-5,0,5,10])
        plt.xticks([-10,0,10])
        # plt.axis('off')
        plt.tick_params(labelsize=15)
        plt.savefig('data3.png', dpi=1000)
        plt.show()

    elif analysis == 'kernel':
       
        # Visualization parameter
        # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

        f, ax = plt.subplots(1)
        sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 3},  color="steelblue", label="Original")
        sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 3, 'linestyle':'--'},  color="darkorange", label="TimeLDM")
        # Plot formatting

        plt.legend(prop={'size': 15})
        # plt.legend()
        plt.xlabel('Data Value', fontsize=18)
        plt.ylabel('Data Density Estimate', fontsize=18)
        # plt.rcParams['pdf.fonttype'] = 42
        plt.xticks([0.4,0.6,0.8])
        plt.yticks([0,1,2,3,4,5])

        # plt.xticks([0.6,0.8,1.0])
        # plt.yticks([1,2,3])
        
        plt.tick_params(labelsize=18)
        # plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        # plt.ylim((0, 12))
        plt.savefig('data4.png', dpi=1000,bbox_inches='tight')
        plt.show()
        plt.close()


def tsne_projection(data, num_classes, color_dict=None, class_dict=None, compare=3000):
   # TSNE anlaysis
   tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
   # Plotting
   f, ax = plt.subplots(1)
   for i in range(num_classes):
      data_class = data[i]
      no, seq_len, dim = data_class.shape
      anal_sample_no = min([compare, data_class.shape[0]])
      idx = np.random.permutation(data_class.shape[0])[:anal_sample_no]
      data_class = data_class[idx]

      for j in range(anal_sample_no):
        if (j == 0):
            prep_data = np.reshape(np.mean(data_class[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(data_class[j, :, :], 1), [1, seq_len])))
            
      colors = [color_dict[i] for j in range(anal_sample_no)]

      tsne_results = tsne.fit_transform(prep_data)
      plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                  c=colors[:], alpha=0.2, label=class_dict[i])
      
   ax.legend()

   plt.title('t-SNE plot')
   plt.xlabel('x-tsne')
   plt.ylabel('y_tsne')
   plt.show()


if __name__ == '__main__':
   class_dict = {0:'StandingUpFS', 1:'StandingUpFL', 2:'Walking', 3:'Running',
                 4:'GoingUpS', 5:'Jumping', 6:'GoingDownS', 7:'LyingDownFS', 8:'SittingDown'}

   color_dict = {0:'lightblue', 1:'lightcoral', 2:'lightcyan', 3:'lightgoldenrodyellow',
                 4:'lightgreen', 5:'lightgray', 6:'lightpink', 7:'lightsalmon', 8:'lightseagreen'}

   num_classes = 9
   ori_data = []
   fake_data = []

   for i in range(num_classes):
       ori_data_i = np.load(f'./OUTPUT/samples/activity_norm_{i}_truth.npy')
       ori_data_i = (ori_data_i + 1) / 2
       ori_data.append(ori_data_i)

       fake_data_i = np.load(f'./OUTPUT/samples/ddpm_fake_activity_{i}.npy')
       fake_data_i = (fake_data_i + 1) / 2
       fake_data.append(fake_data_i)
    
   tsne_projection(ori_data, num_classes, color_dict=color_dict, class_dict=class_dict, compare=12000)