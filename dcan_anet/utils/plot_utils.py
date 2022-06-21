import matplotlib.pyplot as plt
import numpy as np


def plot_result(opt, num_proposals, recall, average_recall, tiou_threshold=np.linspace(0.5, 0.95, 10)):
    """
    Plot result curve.

    Arguements:
        opt: (config): parameters.
        num_proposals: (int[100]): average number of selected proposals per video, which is just range from 1 to 100.
        recall: (float[10][100]): the recall of certain proposal in certain tIoU. 
        average_recall: (float[100]): average recall in certain tIoU, which is equivalent to `recall.mean(axis=0)`.
        tiou_threshold: (np.ndarray[10]): temporal iou threshold.
    """

    # Image setting.
    fn_size = 14
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']

    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    # AR: integration(area below the curve) while x-demension is 'num_proposals' and y-demension is 'recall'.
    AR = np.zeros_like(tiou_threshold)
    for i in range(recall.shape[0]):
        AR[i] = np.trapz(recall[i], num_proposals)

    # Plot a black line which presents AR@AN(Average Recall vs Average Number Of Proposals).
    ax.plot(num_proposals, average_recall, color=colors[0],
            label="tIoU = 0.5:0.05:0.95, area = " + str(int(np.trapz(average_recall, num_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    # Plot 5 lines which present recall in certain tIoU(0.5/0.6/0.7/0.8/0.9).
    for i, tiou in enumerate(tiou_threshold[::2]):
        ax.plot(num_proposals, recall[2 * i, :], color=colors[i + 1],
                label="tIoU = " + str(tiou) + ", area = " + str(int(AR[2 * i] * 100) / 100.),
                linewidth=4, linestyle='--', marker=None)

    # Print details.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right')

    # Save image.
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    # plt.setp(): set property
    # get_xticklabels(): scale mark setting
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    # plt.show()    
    plt.savefig(opt.save_fig_path)
