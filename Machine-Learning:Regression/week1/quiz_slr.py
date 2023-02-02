# Fitting a simple linear regression model on housing data
import turicreate as tc

sales = tc.SFrame("home_data.sframe")
train_data, test_data = sales.random_split(.8, seed=0)

def simple_linear_regression(input_feature, output):
    slope = (sum(input_feature*output) - (sum(input_feature)*sum(output)/len(output))) / (sum(input_feature**2) - (sum(input_feature)*sum(input_feature)/len(input_feature)))
    intercept = sum(output)/len(output) - slope * sum(input_feature)/len(input_feature)
    return(intercept, slope)


result = simple_linear_regression(train_data['sqft_living'], train_data['price'])
squarefeet_intercept = result[0]
squarefeet_slope = result[1]

print("Intercept is {:.2f}.".format(squarefeet_intercept))
print("Slope is {:.2f}.".format(squarefeet_slope))

def get_regression_predications(input_feature, intercept, slope):
    predicated_output = intercept + slope*input_feature
    return(predicated_output)


output = get_regression_predications(2650, squarefeet_intercept, squarefeet_slope)
print("Predicted output is {:.2f}." .format(output))