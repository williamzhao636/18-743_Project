import sys
import numpy as np

# Convert to dictionary for efficiency
def file_to_dict(file_name):
    # Open files to get data from
    fileData = open(file_name, "r")
    startInd = 0
    currInput = fileData.readline()
    retDict = {}
    revDict = {}

    while (currInput != ""):
        currInput = currInput.strip()
        retDict[currInput] = startInd
        revDict[startInd] = currInput
        startInd += 1
        currInput = fileData.readline()

    # Close files that were opened
    fileData.close()

    return (retDict, revDict, startInd)

# Convert validation_input to numbers
def convert_num(file_name, ind_to_word, ind_to_tag):
     # Open files to write to
    fileData = open(file_name, "r")
    word_ind = []
    tag_ind = []
    currInput = fileData.readline()

    while (currInput != ""):
        currInput = currInput.strip()
        currData = currInput.split(" ")
        res_word_Ind = []
        res_tag_Ind = []
        
        for combin in currData:
            currVal = combin.split("_")
            res_word_Ind.append(ind_to_word[currVal[0]])
            res_tag_Ind.append(ind_to_tag[currVal[1]])

        word_ind.append(res_word_Ind)
        tag_ind.append(res_tag_Ind)
        currInput = fileData.readline()
    
    # Close files that were opened
    fileData.close()

    return (word_ind, tag_ind)
'''
# Possible change: Store as logs, predict with logs
def forward(words_line, prior, emit, trans):
    alpha_L = {}
    f_word = words_line[0] - 1
    alpha = np.multiply(np.reshape(emit[:,f_word], (num_tag, 1)), prior)
    alpha_L[0] = alpha

    for i in range(len(words_line[1:])):
        word = words_line[i+1] - 1
        alpha = np.multiply(np.reshape(emit[:,word], (num_tag, 1)), np.transpose(trans) @ alpha)
        alpha_L[i+1] = alpha
    return alpha_L
'''
def forward(words_line, prior, emit, trans):
    alpha_L = {}
    f_word = words_line[0]
    #alpha = np.multiply(np.reshape(emit[:,f_word], (num_tag, 1)), prior)
    alpha = np.log(np.multiply(np.reshape(emit[:,f_word], (num_tag, 1)), prior))
    alpha_L[0] = alpha

    for i in range(len(words_line[1:])):
        word = words_line[i+1]
        recurse = np.reshape(np.array([0.0] * len(trans)), (len(trans), 1))
        for j in range(len(trans)):
            recurse[j] = log_trick(alpha + np.log(np.reshape(trans[:,j], (num_tag, 1))))
        alpha = np.log(np.reshape(emit[:,word], (num_tag, 1))) + recurse
        #alpha = np.multiply(np.reshape(emit[:,word], (num_tag, 1)), np.transpose(trans) @ alpha)
        alpha_L[i+1] = alpha
    return alpha_L

def backward(words_line, emit, trans, num_tag):
    beta_L = {}
    beta = np.log(np.reshape(np.array([1.0] * num_tag), (num_tag, 1)))
    #beta = np.reshape(np.array([1.0] * num_tag), (num_tag, 1))
    beta_L[len(words_line[:-1])] = beta
    for i in range(len(words_line[:-1])-1, -1, -1):
        word = words_line[i+1]
        recurse = np.reshape(np.array([0.0] * len(trans)), (len(trans), 1))
        for j in range(len(trans)):
            recurse[j] = log_trick(beta + np.log(np.reshape(emit[:,word], (num_tag, 1))) + np.log(np.reshape(trans[j], (num_tag, 1))))
        beta = recurse
        #beta = trans @ np.multiply(np.reshape(emit[:,word], (num_tag, 1)), beta)
        beta_L[i] = beta
    return beta_L

def predict(words_line, tags_ind, alpha_L, beta_L):
    tags_L = []
    cap_t = len(words_line)
    num_correct = 0
    num_count = 0
    for i in range(cap_t):
        tag = np.argmax(np.multiply(np.exp(alpha_L[i]), np.exp(beta_L[i])))
        #print(np.multiply(alpha_L[i], beta_L[i]))
        #tag = np.argmax(np.multiply(alpha_L[i], beta_L[i]))
        tags_L.append(tag)
        if tag == tags_ind[i]:
            num_correct += 1
        num_count += 1
    return tags_L, num_correct, num_count

def convert(words, tags, rev_words, rev_tags):
    ret_S = ""
    for i in range(len(words)):
        ret_S += rev_words[words[i]] + "_" + rev_tags[tags[i]] + " "
    ret_S = ret_S[:-1] + '\n'
    return ret_S

def hmm(words, tags, prior, emit, trans, num_tag, rev_words, rev_tags):
    result = ""
    avg_logl = 0
    count = 0
    num_count_tag = 0
    num_count_correct = 0
    for i in range(len(words)):
        line = words[i]
        line_tags = tags[i]
        count += 1
        alpha_D = forward(line, prior, emit, trans)
        beta_D = backward(line, emit, trans, num_tag)
        predictions, num_correct, num_count = predict(line, line_tags, alpha_D, beta_D)
        print(predictions)
        num_count_tag += num_count
        num_count_correct += num_correct
        log_likey = log_likeli(alpha_D, len(line))
        avg_logl += log_likey
        result += convert(line, predictions, rev_words, rev_tags)
    accuracy = num_count_correct / num_count_tag
    avg_logl = avg_logl / count
    return result, avg_logl, accuracy

def log_likeli(alpha_D, length):
    #print(alpha_D)
    last_alpha = alpha_D[length-1]
    #return np.log(np.sum(last_alpha))
    #print(np.log(np.sum(np.exp(last_alpha))))
    return log_trick(last_alpha)

def log_trick(vector):
    m = np.max(vector)
    return m + np.log(np.sum(np.exp(vector - m)))

def write_predictions(file_name, predictions):
    # Open files to write to
    fileData = open(file_name, "w")

    fileData.write(predictions)
    
    # Close files that were opened
    fileData.close()

def write_metrics(file_name, avg_log_l, accuracy):
    # Open files to write to
    fileData = open(file_name, "w")
    toWrite = "Average Log-Likelihood: %f\nAccuracy: %f\n" % (avg_log_l, accuracy)

    fileData.write(toWrite)
    
    # Close files that were opened
    fileData.close()

if __name__ == '__main__':
    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    mectric_file = sys.argv[8]

    word_dict, rev_word, num_word = file_to_dict(index_to_word)
    tag_dict, rev_tag, num_tag = file_to_dict(index_to_tag)
    word_ind, tag_ind = convert_num(validation_input, word_dict, tag_dict)

    prior = np.fromfile(hmmprior, dtype=float, sep=" ")
    prior = np.reshape(prior, (num_tag, 1))
    
    emit = np.fromfile(hmmemit, dtype=float, sep=" ")
    emit = np.reshape(emit, (num_tag, num_word))

    trans = np.fromfile(hmmtrans, dtype=float, sep=" ")
    trans = np.reshape(trans, (num_tag, num_tag))

    predictions, avg_log_l, accuracy = hmm(word_ind, tag_ind, prior, emit, trans, num_tag, rev_word, rev_tag)
    write_predictions(predicted_file, predictions)
    write_metrics(mectric_file, avg_log_l, accuracy)