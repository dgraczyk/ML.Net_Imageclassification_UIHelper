using ML.Net_Imageclassification.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Net_Imageclassification_UI
{
    public class PredictionResult
    {
        public string ImageSource { get; }

        public string Prediction { get; }

        public string Score { get; }

        public PredictionResult(ModelInput modelInput, ModelOutput modelOutput, List<string> predictionLabels)
        {
            this.ImageSource = modelInput.ImageSource;
            this.Prediction = modelOutput.Prediction;

            var sb = new StringBuilder();

            for (int i = 0; i < predictionLabels.Count; i++)
            {
                sb.Append($"{predictionLabels[i]}: {Math.Round(modelOutput.Score[i] * 100)}%\n");
            }

            this.Score = sb.ToString();
        }
    }
}