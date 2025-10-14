# AJDANN Frontend Application

## 🎯 Overview
This is a beautiful, modern web frontend for the AJDANN (Advanced Neural Network) survival prediction system. It provides an intuitive interface for healthcare professionals to input patient data and receive AI-powered survival predictions.

## ✨ Features
- **Modern UI Design**: Clean, professional interface with gradient backgrounds and smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Real-time Predictions**: Instant AI-powered survival predictions
- **Input Validation**: Ensures all data is properly formatted before submission
- **Sample Data**: Quick test buttons with pre-filled sample cases
- **Visual Results**: Clear presentation of mPFS and PFS6 predictions

## 🚀 Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete application
python start_app.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (if not already done)
python ajdANN_v6a.py

# 3. Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 4. Open the frontend
# Simply open index.html in your web browser
```

## 📱 How to Use

1. **Open the Application**: Navigate to `http://localhost:8000` in your web browser
2. **Enter Patient Data**: Fill in all required fields:
   - **Age**: Patient's age in years (18-120)
   - **Gender**: Male or Female
   - **Resection Status**: Percentage of tumor resection (0-100%)
   - **Karnofsky Score**: Performance status score (0-100%)
   - **Methylation Status**: Methylation level percentage (0-100%)
   - **Pre-Treatment History**: Level of previous treatment (None to Extensive)
3. **Get Predictions**: Click "Predict Survival" to receive AI-generated predictions
4. **View Results**: See predicted mPFS (months) and PFS6 (probability %)

## 🎨 Design Features

### Color Scheme
- **Primary Gradient**: Purple to blue gradient (#667eea to #764ba2)
- **Success Colors**: Green gradient for positive actions
- **Background**: Gradient background with glass-morphism effects
- **Text**: High contrast for readability

### UI Elements
- **Glass-morphism Cards**: Semi-transparent cards with blur effects
- **Smooth Animations**: Fade-in effects and hover transitions
- **Icon Integration**: Font Awesome icons throughout the interface
- **Responsive Grid**: Adapts to different screen sizes

## 🔧 Technical Details

### Frontend Technologies
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with Flexbox and Grid
- **JavaScript**: Vanilla JS for API communication
- **Font Awesome**: Icons
- **Google Fonts**: Inter font family

### API Integration
- **Endpoint**: `http://localhost:8000/predict`
- **Method**: POST
- **Data Format**: JSON
- **Response**: mPFS and PFS6 predictions

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🧪 Testing

The application includes sample data buttons for quick testing:
- **Sample Case 1**: Elderly male with extensive history
- **Sample Case 2**: Middle-aged male with moderate history  
- **Sample Case 3**: Young female with minimal history

## 📊 Input Parameters Explained

| Parameter | Description | Range |
|-----------|-------------|-------|
| Age | Patient's chronological age | 18-120 years |
| Gender | Biological gender | Male (1) / Female (0) |
| Resection Status | Percentage of tumor removed | 0-100% |
| Karnofsky Score | Performance status measure | 0-100% |
| Methylation Status | DNA methylation level | 0-100% |
| Pre-Treatment History | Previous treatment level | 1-4 (None to Extensive) |

## 🎯 Output Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| mPFS | Median Progression-Free Survival | Months |
| PFS6 | 6-Month Progression-Free Survival | Probability (%) |

## 🔒 Security Notes

- The application runs locally and doesn't store patient data
- All predictions are made in real-time
- No data is transmitted to external servers
- CORS is enabled for local development

## 🐛 Troubleshooting

### Common Issues
1. **API Connection Error**: Make sure the FastAPI server is running on port 8000
2. **Model Loading Error**: Ensure the model files exist in `saved_models_v6a/`
3. **Dataset Not Found**: Check that the CSV file path in `ajdANN_v6a.py` is correct

### Error Messages
- **"API Error"**: Server is not running or there's a network issue
- **"Please fill in all fields"**: Validation error - check input values
- **"Model training failed"**: Dataset file missing or corrupted

## 📞 Support

For technical support or questions about the application, please refer to the main project documentation or contact the development team.

---

**Note**: This application is designed for research and educational purposes. Always consult with healthcare professionals for clinical decisions. 