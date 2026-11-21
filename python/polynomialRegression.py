import numpy as np

def get_poly_terms(datum, max_deg):
    if datum.size==1:
        x = np.array([datum[0]**deg for deg in range(max_deg+1)])
        return x

    poly_terms = np.array([])
    for deg in range(max_deg+1):
        poly_terms = np.append(poly_terms, datum[0]**deg*get_poly_terms(datum[1:], max_deg-deg))

    return poly_terms

def get_design_matrix(data, max_deg):
    num_columns = get_poly_terms(data[0,:], max_deg).size
    X = np.ones((data.shape[0], num_columns))

    for i in range(data.shape[0]):
        X[i,:] = get_poly_terms(data[i,:], max_deg)
    return X

def posteriors(design_matrix, prior_lambda, prior_mu, beta_hat):
    posterior_lambda = np.matmul(np.transpose(design_matrix), design_matrix) + prior_lambda
    posterior_mu = np.matmul(np.linalg.inv(posterior_lambda), (np.matmul(np.matmul(np.transpose(design_matrix), design_matrix), beta_hat) + np.matmul(prior_lambda, prior_mu)))
    return posterior_mu, posterior_lambda

def beta_hat(X, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)

def blr(x, y, lamda, mu, max_deg, batch_size=0):
    num_data_points = y.size
    
    if batch_size == 0:
        batch_size = num_data_points

    for i in range(num_data_points//batch_size):
        X = get_design_matrix(np.array(x[i*batch_size:(i+1)*batch_size]), max_deg)
        betahat = beta_hat(X, np.array(y[i*batch_size:(i+1)*batch_size]))
        mu, lamda = posteriors(X, lamda, mu, betahat)

    return mu, lamda

