### Importing libraries ###

import torch
import torch.nn as nn
import torch.nn.functional as F


### TNN Column Layer ###
## This class models all the columns in a single layer based on the input size, kernel (or RF) size and stride.
## Each type of data, e.g. input spiketimes, weights, etc., is modeled as a corresponding single tensor.
## The key function of this layer is to take in input spiketimes, calculate excitatory spiketimes for all neurons
## (in all columns) in the layer and perform 1-WTA lateral inhibition on those excitatory spiketimes within each
## column in the layer separately.
## No spike is represented as having a spiketime of inifnity, i.e., "float('Inf')".
##
## __init__ function initializes all the variables and weights associated with the column layer and is called only
## once when the class is instantiated.
## __call__ function is automatically invoked whenever data is passed into the layer.
##
## Arguments for __init__:
##   1) inputsize - a tuple of (height, width) of the input feature map. E.g. For a 28x28 image, it will be (28,28).
##   2) rfsize    - a tuple of (height, width) of the kernel filter. E.g. For a 3x3 filter, it will be (3,3).
##   3) stride    - stride by which the kernel slides across the input feature map.
##   4) nprev     - number of channels in the input image (for first column layer) or number of neurons per column
##                  in the previous column layer (if the current column layer is not the very first column layer).
##   5) q         - number of neurons per column in the current column layer.
##   6) theta     - spiking threshold for all neurons in the current column layer.
##   7) wres      - bit resolution for weights (default=3).
##   8) w_init    - weight initialization mode (default=half). 4 modes available (see comments below).
##   9) ntype     - response function type (default=rnl). Selects between step-no-leak (snl) and ramp-no-leak (rnl).
##   10) device   - device type (default=cpu). Used for CUDA support to run on GPUs if needed.
##
## Key Class Variables:
##   1) rows          - number of rows in the output 2D map after sliding the kernel filter across the input 2D map.
##   2) cols          - number of columns (NOT TNN COLUMNS) in the output 2D map after sliding the kernel filter
##                      across the input 2D map.
##   3) p             - number of synapses (or equivalently weights) per neuron in the current layer.
##   4) q             - number of neurons per column.
##   5) num           - total number of neurons in the current column layer.
##   6) wmax          - maximum weight value given wres-bit weights. E.g. wmax=7 for 3-bit weights.
##   7) weights       - stores all the weights in the current column layer. num*p gives the total number of weights.
##                      modeled as a 2D tensor with dimensions (num, p).
##   8) ec_spiketimes - excitatory column spiketimes for all neurons in the layer BEFORE 1-WTA LI.
##                      Its shape is (rows*cols, q).
##                      Separating out the neurons per column dimension is helpful while performing LI.
##   9) li_spiketimes - excitatory column spiketimes for all neurons in the layer AFTER 1-WTA LI.
##                      Its shape is (rows*cols, q).
##
## Argument for __call__:
##   data - 3D tensor of input spiketimes with shape (inputsize[0], inputsize[1], nprev). Spiketime can be either
##          "float('Inf')" or any integer value. Note, the code should be generic enough to accept arbitrary
##          integer spiketimes, and not restrict to PosNeg case (which can only take values, 0 or float('Inf')).
##          Note, input "2D" map as mentioned earlier in comments was simply referring to the first two out of the
##          three dimensions. The first two dimensions represent the 2D spatial distribution and the third dimension
##          can be imagined as adding a depth to it.
##          E.g., if the input is coming from a PosNeg encoded 28x28 image, then its shape would be (28,28,2).
##          Else, if the input is coming from a previous column layer which had 5x5=25 columns arranged in a 5x5
##          tile format and 10 neurons per column, the input data shape would be (5,5,10).
##
## __call__ function returns:
##   1) output_spiketimes_nextlayer - output spiketimes after LI to be passed onto the next column layer. It is
##                                    appropriately shaped as (rows, cols, q). Note that 'q' acts as 'nprev' and
##                                    '(rows,cols)' acts as 'inputsize' for the next column layer.
##   2) input_spiketimes            - it's passed as output, so it can be later on used as input to STDP. It is
##                                    appropriately reshaped as (nums, p) to match the shape of weights.
##   3) output_spiketimes_stdp      - output spiketimes after LI to be passed as input to STDP. It is
##                                    appropriately reshaped as (nums, p) to match the shape of weights. Note, all
##                                    synapses for a particular neuron see the same output spiketime of that neuron.


class TNNColumnLayer(nn.Module):
    def __init__(self, inputsize, rfsize, stride, nprev, q, theta, wres=3, w_init="half", ntype="rnl",                  device="cpu"):
        super(TNNColumnLayer, self).__init__()

        # Automatically convert to tuple if only a single value is provided
        if not isinstance(inputsize,tuple):
            inputsize      = (inputsize,inputsize)
        if not isinstance(rfsize,tuple):
            rfsize         = (rfsize,rfsize)

        self.rfsize        = rfsize
        self.stride        = stride

        # self.rows * self.cols gives the total number of RFs (equiavalently the total number of TNN columns) in
        # the current layer.
        # Note that each RF maps to a single TNN column.
        #self.rows          = ((inputsize[0]-rfsize[0])//stride) + 1
        #self.cols          = ((inputsize[1]-rfsize[1])//stride) + 1
        self.rows = 1
        self.cols = 1

        # Note that each input in an nxn RF goes to all neurons within the corresponding column and each RF input
        # has nprev values/channels. Therefore each neuron gets n^2*nprev inputs resulting in as many synapses.
        # self.p             = rfsize[0]*rfsize[1]*nprev
        self.p             = 12

        # Number of neurons per column
        self.q             = q

        # Total number of neurons in the current layer
        self.num           = self.rows*self.cols*self.q

        self.theta         = theta
        self.wres          = wres
        self.ntype         = ntype
        self.device        = device

        self.wmax          = 2**self.wres - 1

        # Synaptic weight initialization. 4 modes available.
        # zero - initializes all weights to zero.
        # half - initializes all weights to wmax/2.
        # uniform - initializes weights to integers uniformly distributed between 0 and wmax.
        # normal - initializes weights to normally distributed values centered around wmax/2.
        if w_init       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.num, self.p), requires_grad=False)
        elif w_init     == "half":
            self.weights   = nn.Parameter((self.wmax/2)*torch.ones(self.num, self.p), requires_grad=False)
        elif w_init     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low=0, high=self.wmax+1, size=(self.num, self.p)).type(torch.FloatTensor), requires_grad=False)
        elif w_init     == "normal":
            self.weights   = nn.Parameter(torch.round(((self.wmax+1)/2+torch.randn(self.num, self.p)).clamp_(0,self.wmax)), requires_grad=False)

        # Initialize the unsupervised STDP class for the current column layer
        self.stdp          = STDP_Deterministic(wres)

        # Initialize miscellaneous variables here.
        self.constidx      = torch.arange(self.num).unsqueeze(1).repeat(1, self.p).to(device)
        self.constdec      = torch.arange(self.p, 0, -1).repeat(self.num,1).to(device)
        self.constidx2     = torch.arange(self.num).to(device)
        self.ones          = torch.ones(self.rows*self.cols,self.q).to(device)

        ### WRITE YOUR CODE BELOW TO INITIALIZE ANY VARIABLES YOU LIKE (OPTIONAL) ###
        self.constw        = torch.arange(self.wmax).repeat(self.num,1).to(device)
        self.constw2       = torch.arange(self.wmax, 0, -1).repeat(self.num,1).to(device)
        self.mones         = -1.0 * torch.ones(self.rows*self.cols,self.q).to(device)


    def __call__(self, data):

        ######################################### NO NEED TO MODIFY ##############################################

        ### Generate all the RFs from input data ###
        # Note, all neurons within a particular neuron see the same set of input spiketimes.
        # Finally, it's reshaped to (nums, p) to match the shape of weights.

        sliced_data                              = data.unfold(0, self.rfsize[0], self.stride).unfold(1, 2 ,self.stride)
        #sliced_data                              = data.unfold(0, self.rfsize[0], self.stride)

        input_spiketimes                         = sliced_data.unsqueeze(2).repeat(1,1,self.q,3,1,1).reshape(self.num,self.p)

        ### Simulate SNL response function and calculate SNL output spiketimes for all neurons in the layer ###
        # "ec_times" holds the excitatory column spiketimes before LI for all neurons in the layer.
        # It is a 1D tensor with shape (num).
        # "self.pot" holds the body potentials of all neurons in the layer at the time of their spiking. This is
        # later used by 1-WTA lateral inhibition to break the ties.

        eff_weights                              = self.weights.clone()
        nullinput_idx                            = input_spiketimes == float('Inf')
        eff_weights[nullinput_idx]               = 0

        sorted_times, sorted_idx                 = torch.sort(input_spiketimes, dim=1)
        sorted_weights                           = eff_weights[self.constidx, sorted_idx]

        temp_pot                                 = torch.cumsum(sorted_weights, dim=1)
        tmp_pot2                                 = temp_pot.clone()
        temp_pot                                 = torch.where(temp_pot<self.theta, 0, 1)

        idxmul                                   = temp_pot * self.constdec
        indices                                  = torch.argmax(idxmul, 1)

        ec_times                                 = sorted_times[self.constidx2, indices]
        nullspike_idx                            = (torch.sum(temp_pot, dim=1) == 0)
        ec_times[nullspike_idx]                  = float('Inf')

        self.pot                                 = tmp_pot2[self.constidx2, indices]

        ##########################################################################################################

        ### Simulate RNL response function and calculate RNL output spiketimes for all neurons in the layer ###
        # "ec_times" holds the excitatory column spiketimes before LI for all neurons in the layer.
        # It is a 1D tensor with shape (num).
        # "self.pot" holds the body potentials of all neurons in the layer at the time of their spiking. This is
        # later used by 1-WTA lateral inhibition to break the ties.

        if self.ntype == "rnl":
            ##################################### WRITE YOUR CODE BELOW ##########################################

            # TASK 1: Modify "ec_times" variable to hold the RNL spiketimes.
            # TASK 2: Modify "self.pot" variable to hold the neuron body potentials at the time of spiking, as per
            #         RNL response function.
            #
            # You only need to fill within this "IF" condition. But you are allowed to add your code outside the
            # "IF" condition if needed, as long as you don't break the SNL code and model correct functionality.
            #
            # Hint 1: You can use the SNL variables as a starting point, e.g. ec_times, nullspike_idx, self.pot.
            # Hint 2: Given a set of input spiketimes and weights, can RNL response function generate an output
            #         spike, if the SNL response function didn't?
            # Hint 3: Useful PyTorch functions: unsqueeze, repeat, sum, argmax, clone
            #         Note, it's not an exhaustive list! And you don't have to necessarily use these functions.

            nonnull_idx                          = torch.logical_not(nullspike_idx)
            rnl_times_orig                       = ec_times[nonnull_idx]
            pruned_num                           = rnl_times_orig.shape[0]

            if pruned_num != 0:
                rnl_times                            = rnl_times_orig.unsqueeze(1).repeat(1,self.wmax) + self.constw[0:pruned_num]
                rnl_times                            = rnl_times.unsqueeze(2).repeat(1,1,self.p)

                in_times                             = input_spiketimes[nonnull_idx].unsqueeze(1).repeat(1,self.wmax,1)
                resp_unnorm                          = rnl_times + 1 - in_times
                resp_weights                         = self.weights[nonnull_idx].unsqueeze(1).repeat(1,self.wmax,1)
                resp                                 = torch.minimum(resp_unnorm, resp_weights)
                resp                                 = resp.clamp(min=0)
                respsum                              = torch.sum(resp, dim=2)
                temp_resp                            = respsum.clone()

                lt_resp                              = respsum  < self.theta
                respsum[lt_resp]                     = 0
                respsum[torch.logical_not(lt_resp)]  = 1

                idxmul2                              = respsum * self.constw2[0:pruned_num]
                indices                              = torch.argmax(idxmul2, 1)
                ec_times[nonnull_idx]                = rnl_times_orig + indices

                self.pot[nonnull_idx]                = temp_resp[self.constidx2[0:pruned_num], indices]

            #####################################################################################################

        ######################################### NO NEED TO MODIFY #############################################
        self.ec_spiketimes                       = ec_times.reshape(-1,self.q)
        self.pot                                 = self.pot.reshape(-1,self.q)

        self.li_spiketimes                       = float('Inf')*self.ones
        #########################################################################################################

        #### Simulate 1-WTA Lateral Inhibition and calculate output spiketimes for all neurons in the layer #####
        # Only 1 neuron within a column is allowed to spike, other spiketimes within that column are changed to
        # float('Inf'). And this has to be done for all columns in the layer.
        # Tie breaking is done by selecting the neuron with the largest body potential at the time of spiking.

        ##################################### WRITE YOUR CODE BELOW #############################################
        # TASK: Modify "self.li_spiketimes" variable to hold the excitatory column spiketimes for all neurons
        #       in the layer after 1-WTA LI. Implement tie breaking by selecting the neuron with the largest body
        #       potential at the time of spiking.
        #
        # Hint: Useful PyTorch functions: unsqueeze, repeat, min, scatter.
        #       Note, it's not an exhaustive list! And you don't have to necessarily use these functions.

        minv, _                                  = torch.min(self.ec_spiketimes, dim=1)
        minvr                                    = minv.unsqueeze(1).repeat(1,self.q)
        idcs                                     = torch.nonzero(self.ec_spiketimes == minvr, as_tuple=True)
        modpot                                   = self.mones.clone()
        modpot[idcs]                             = self.pot[idcs]
        _, iid                                   = torch.max(modpot, dim=1)
        self.li_spiketimes.scatter_(1, iid.unsqueeze(1), minv.unsqueeze(1))

        #########################################################################################################

        ######################################### NO NEED TO MODIFY #############################################
        output_spiketimes_stdp                   = self.li_spiketimes.reshape(-1).unsqueeze(1).repeat(1,self.p)
        output_spiketimes_nextlayer              = self.li_spiketimes.reshape(self.rows,self.cols,self.q)

        return output_spiketimes_nextlayer, input_spiketimes, output_spiketimes_stdp
        #########################################################################################################


### Determinsitic STDP algorithm ###
## This class models the deterministic STDP cases (see recitation and handout for more details)

## __init__ function initializes wmax and is invoked during class instantiation.
## __call__ function is automatically invoked whenever data is passed into the instantiated class.
##
## Argument for __init__:
##   wres       - bit resolution for weights (default=3).
##
## Arguments for __call__:
##   1) ec_in   - input spiketime tensor with shape (num, p).
##   2) li_out  - excitatory output spiketime tensor (after LI) with shape (num, p).
##   3) weights - all synaptic weights in the layer with shape (num, p).
##   3) capture - "mu_plus" parameter used for incrementing the weights in Case 1.
##   3) search  - "mu_plus" parameter used for incrmeenting the weights in Case 3.
##   3) backoff - "mu_minus" parameter used for decrementing the weights in Case 2 and Case 4.
##
## __call__ function returns:
##   weights - updated weights after applying determinsitic STDP learning with shape (num, p).

class STDP_Deterministic():
    def __init__(self, wres):
        self.wmax           = 2**(wres)-1

        ### WRITE YOUR CODE BELOW TO INITIALIZE ANY VARIABLES YOU LIKE (OPTIONAL) ###

        #############################################################################

    def __call__(self, ec_in, li_out, weights, capture, search, backoff):

        ##### Performs STDP weight update based on correspondig input/output spiketimes and current weights #####

        ##################################### WRITE YOUR CODE BELOW #############################################
        # TASK: Modify "weights" variable to hold the updated weights after applying all four STDP cases.
        #
        # Hint 1: Generate the conditions for each case separately and apply appropriate weight updates. Note, all
        #       four conditions are mutually exclusive.
        # Hint 2: Make sure you manually clamp the weights between 0 and wmax at the end.

        # Case 1 (capture)
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights>=self.wmax/2)] += capture*capture
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights<self.wmax/2)] += capture*capture/2

        # Case 2 (minus)
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights<self.wmax/2)] -= backoff*backoff
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights>=self.wmax/2)] -= backoff*backoff/2

        # Case 3 (search)
        weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] += search

        # Case 4 (backoff)
        weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights<self.wmax/2)] -= backoff*backoff
        weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights>=self.wmax/2)] -= backoff*backoff/2

        weights = weights.clamp(0, self.wmax)

        #########################################################################################################

        ######################################### NO NEED TO MODIFY #############################################
        return nn.Parameter(weights, requires_grad=False)
        #########################################################################################################
