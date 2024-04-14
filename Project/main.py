import utils.data_loader as data_loader
import models.classifier as classifier

def main():
    df = data_loader.load_data('./data/BMW_Vehicles_Data.csv')
    X, y = data_loader.prepare_features(df)
    
    classifier.train_evaluate(X, y)

if __name__ == '__main__':
    main()
  
