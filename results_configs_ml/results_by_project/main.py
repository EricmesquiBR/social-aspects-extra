# script that summarizes all the data from the ML models and transforms it into a dataframe
import pandas as pd
import os, sys, json, re, collections
import matplotlib.pyplot as plt
import seaborn as sns


class Dataframe:
    # best_params, performance
    def __init__(self, feature_importance, performance, config_directory):
        self.feature_importance = feature_importance
        self.performance = performance
        self.config_directory = config_directory

    def calc_precision_by_project_and_smell(self):
        df = pd.read_csv(self.performance)

        # Group by 'project', 'type' and 'model', then get the max 'test_f1' for each group
        grouped = df.groupby(['project', 'type', 'model'])['test_precision'].max()

        return grouped.reset_index()

    def calc_recall_by_project_and_smell(self):
        df = pd.read_csv(self.performance)

        # Group by 'project', 'type' and 'model', then get the max 'test_f1' for each group
        grouped = df.groupby(['project', 'type', 'model'])['test_recall'].max()

        return grouped.reset_index()

    def calc_performance_by_project_and_smell(self):
        # Read the performance csv file
        df = pd.read_csv(self.performance)

        # Group by 'project', 'type' and 'model', then get the max 'test_f1' for each group
        grouped = df.groupby(['project', 'type', 'model'])['test_f1'].max()

        return grouped.reset_index()

    def calc_accuracy_by_project_and_smell(self):
        df = pd.read_csv(self.performance)

        # Group by 'project', 'type' and 'model', then get the max 'test_accuracy' for each group
        grouped = df.groupby(['project', 'type', 'model'])['test_accuracy'].max()

        return grouped.reset_index()

    def calc_feature_frequency(self):
        pass

    def run_rq1(self):
        # table with all perfomances by projects and smell, and
        dataframe1 = self.calc_precision_by_project_and_smell()
        dataframe2 = self.calc_recall_by_project_and_smell()
        dataframe3 = self.calc_performance_by_project_and_smell()
        dataframe4 = self.calc_accuracy_by_project_and_smell()

        # Merge the four datasets
        merged = pd.merge(dataframe1, dataframe2, on=['project', 'type', 'model'])
        merged = pd.merge(merged, dataframe3, on=['project', 'type', 'model'])
        merged = pd.merge(merged, dataframe4, on=['project', 'type', 'model'])

        return merged

    def run_rq2(self):
        # table with all perfomances by projects and smell, and
        dataframe1 = self.calc_performance_by_project_and_smell()
        dataframe2 = self.calc_accuracy_by_project_and_smell()

        # Merge the two datasets
        merged = pd.merge(dataframe1, dataframe2, on=['project', 'type', 'model'])

        return merged

    def run_rq3(self):
        pass


class DataframePlotter:
    def __init__(self, df, output_directory):
        self.df = df
        self.output_directory = output_directory

    def plot_best_performance_by_project(self):
        # Select the top 3 models based on 'test_f1'
        top_models = self.df.sort_values('test_f1', ascending=False)['model'].unique()[:3]

        # Filter the DataFrame to include only the top 3 models
        df_top_models = self.df[self.df['model'].isin(top_models)]

        # Plot the 'test_f1' for each model by project
        plt.figure(figsize=(15, 10))
        sns.barplot(data=df_top_models, x='project', y='test_f1', hue='model', errorbar=None)
        plt.title('Best Performance by Project')
        plt.ylabel('F1 Score')
        plt.xlabel('Project')
        plt.legend(title='Model')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, 'best_performance_by_project.png'))
        plt.close()

    def plot_best_accuracy_by_project(self):
        # Select the top 3 models based on 'test_accuracy'
        top_models = self.df.sort_values('test_accuracy', ascending=False)['model'].unique()[:3]

        # Filter the DataFrame to include only the top 3 models
        df_top_models = self.df[self.df['model'].isin(top_models)]

        # Plot the 'test_accuracy' for each model by project
        plt.figure(figsize=(15, 10))
        sns.barplot(data=df_top_models, x='project', y='test_accuracy', hue='model', errorbar=None)
        plt.title('Accuracy by Project')
        plt.ylabel('Accuracy')
        plt.xlabel('Project')
        plt.legend(title='Model')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, 'accuracy_by_project.png'))
        plt.close()

    def run(self):
        self.plot_best_performance_by_project()
        self.plot_best_accuracy_by_project()


if __name__ == "__main__":

    project_root = sys.path[0]
    all_directories = [x[0] for x in os.walk(project_root)]
    regex = re.compile(r'config\d+')
    config_directories = [directory for directory in all_directories if regex.search(os.path.basename(directory))]
    # Print the list of directories named config{number}
    # config_directories = [
    #     'config1_all_communication_dynamics_all_features'
    # ]
    for directory in config_directories:
        print(directory)
        feature_importance_json = None
        best_params_csv = None
        performance_csv = None
        for file in os.listdir(directory):
            if file.endswith(".json"):
                feature_importance_json = os.path.join((directory), file)
        for file in os.listdir(directory):
            if file.endswith("performance.csv"):
                performance_csv = os.path.join((directory), file)
        dataframe_rq1 = Dataframe(feature_importance_json, performance_csv, directory).run_rq1()
        dataframe_rq1.to_csv(os.path.join((directory), 'rq1.csv'), index=False)