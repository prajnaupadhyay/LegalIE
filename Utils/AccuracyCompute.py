def load_labels(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                test_labels.append(line)
                #print(line)
            
    return np.array(test_labels)
def load_predictions(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Prediction'):
                line = line.strip("Prediction:")
                predictions.append(line)
                
                #print(predictions)
                
    return np.array(predictions)
  
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
predictions = []
test_labels = []
predictions = load_predictions("prediction_ptb_Level.txt")
test_labels = load_labels("ptb_test.txt")
# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)

# Calculate F1 score
f1 = f1_score(test_labels, predictions, average='micro')  # You can choose 'micro', 'macro', 'weighted', or None

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

#Results obtained
#Accuracy: 0.0000
#F1 Score: 0.0000
