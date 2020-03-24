using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.WindowsAPICodePack.Dialogs;
using ML.Net_Imageclassification.Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;

namespace ML.Net_Imageclassification_UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        private string imageFolder;

        public string ImageFolder
        {
            get { return imageFolder; }
            set
            {
                imageFolder = value;
                this.OnPropertyChanged(nameof(ImageFolder));
            }
        }

        private int progressBarMaxValue;

        public int ProgressBarMaxValue
        {
            get { return progressBarMaxValue; }
            set
            {
                progressBarMaxValue = value;
                this.OnPropertyChanged(nameof(ProgressBarMaxValue));
            }
        }

        private int progressBarCurrentValue;

        public int ProgressBarCurrentValue
        {
            get { return progressBarCurrentValue; }
            set
            {
                progressBarCurrentValue = value;
                this.OnPropertyChanged(nameof(ProgressBarCurrentValue));
            }
        }

        private string message;

        public string Message
        {
            get
            {
                return this.message;
            }
            set
            {
                this.message = value;
                this.OnPropertyChanged(nameof(Message));
            }
        }

        private readonly PredictionEngine<ModelInput, ModelOutput> predictionEngine;
        private readonly string modelPath = "MLModel.zip";
        private readonly List<string> predictionLabels;

        public MainWindow()
        {
            InitializeComponent();

            this.DataContext = this;

            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            predictionLabels = this.GetLabelsNames(predictionEngine.OutputSchema, "Score");
        }

        private async void Run(object sender, RoutedEventArgs e)
        {
            this.PredictionResults.ItemsSource = null;

            await Task.Run(() =>
            {
                try
                {
                    var items = new List<PredictionResult>();
                    var files = Directory.GetFiles(this.ImageFolder, "*.jpg");

                    this.ProgressBarCurrentValue = 0;
                    this.ProgressBarMaxValue = files.Length;

                    foreach (var file in files)
                    {
                        this.Message = $"File: {this.ProgressBarCurrentValue++} / {this.ProgressBarMaxValue}";

                        var modelInput = new ModelInput { ImageSource = file };
                        var modelOutput = this.predictionEngine.Predict(modelInput);

                        items.Add(new PredictionResult(modelInput, modelOutput, this.predictionLabels));
                    }

                    this.Dispatcher.Invoke(() =>
                    {
                        this.PredictionResults.Visibility = Visibility.Visible;
                        this.PredictionResults.ItemsSource = items;

                        CollectionView view = (CollectionView)CollectionViewSource.GetDefaultView(this.PredictionResults.ItemsSource);
                        PropertyGroupDescription groupDescription = new PropertyGroupDescription("Prediction");
                        view.GroupDescriptions.Add(groupDescription);
                    });
                }
                catch (Exception ex)
                {
                    this.Message = ex.Message;
                }

                this.Message = "Done";
            });
        }

        private void SelectImages(object sender, RoutedEventArgs e)
        {
            using (var dialog = new CommonOpenFileDialog())
            {
                dialog.IsFolderPicker = true;
                dialog.Title = "Select folder with jpg files";

                var result = dialog.ShowDialog();

                if (result == CommonFileDialogResult.Ok)
                {
                    this.ImageFolder = dialog.FileName;
                }
            }
        }

        private async void CopyTo(object sender, RoutedEventArgs e)
        {
            using (var dialog = new CommonOpenFileDialog())
            {
                dialog.IsFolderPicker = true;
                dialog.Title = "Select output folder";

                var result = dialog.ShowDialog();

                if (result == CommonFileDialogResult.Ok)
                {
                    var files = this.PredictionResults.SelectedItems.OfType<PredictionResult>();

                    await Task.Run(() =>
                    {
                        try
                        {
                            this.ProgressBarCurrentValue = 0;
                            this.ProgressBarMaxValue = files.Count();

                            foreach (var file in files)
                            {
                                this.Message = $"File: {this.ProgressBarCurrentValue++} / {this.ProgressBarMaxValue}";
                                File.Copy(file.ImageSource, Path.Combine(dialog.FileName, Path.GetFileName(file.ImageSource)));
                            }
                        }
                        catch (Exception ex)
                        {
                            this.Message = ex.Message;
                        }

                        this.Message = "Done";
                    });
                }
            }
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }

        // https://stackoverflow.com/a/57259240
        private List<string> GetLabelsNames(DataViewSchema schema, string name)
        {
            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                names[num++] = denseValue.ToString();
            }

            return names.ToList();
        }
    }
}