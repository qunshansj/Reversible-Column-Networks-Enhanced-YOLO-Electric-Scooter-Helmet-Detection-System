python

class CKA:
    def __init__(self):
        pass

    def unbiased_HSIC(self, K, L):
        '''Computes an unbiased estimator of HISC. This is equation (2) from the paper'''

        #create the unit **vector** filled with ones
        n = K.shape[0]
        ones = np.ones(shape=(n))

        #fill the diagonal entries with zeros
        np.fill_diagonal(K, val=0) #this is now K_tilde
        np.fill_diagonal(L, val=0) #this is now L_tilde

        #first part in the square brackets
        trace = np.trace(np.dot(K, L))

        #middle part in the square brackets
        nominator1 = np.dot(np.dot(ones.T, K), ones)
        nominator2 = np.dot(np.dot(ones.T, L), ones)
        denominator = (n-1)*(n-2)
        middle = np.dot(nominator1, nominator2) / denominator


        #third part in the square brackets
        multiplier1 = 2/(n-2)
        multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
        last = multiplier1 * multiplier2

        #complete equation
        unbiased_hsic = 1/(n*(n-3)) * (trace + middle - last)

        return unbiased_hsic

    def CKA(self.......
