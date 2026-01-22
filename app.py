"""
Vista API - Virtual Staging with AWS S3 Integration and Versioning
Flask application using Blueprint pattern for clean architecture
"""
from flask import Flask
from flask_cors import CORS
from datetime import datetime

# Import the Blueprints from controllers
from controllers.virtual_staging_controller import virtual_staging_bp
from controllers.aws_s3_controller import aws_s3_bp

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Enable CORS
CORS(app)

# Register the Blueprints
app.register_blueprint(virtual_staging_bp)
app.register_blueprint(aws_s3_bp)


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'Vista Virtual Staging API',
        'timestamp': datetime.utcnow().isoformat(),
        'features': [
            'AWS S3 Integration',
            'Version History',
            'Save/Revert Changes',
            'Image Generation with Gemini',
            'Refinement Support'
        ]
    }, 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {'error': 'Resource not found'}, 404


@app.errorhandler(405)
def method_not_allowed(error):
    return {'error': 'Method not allowed'}, 405


@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Vista Virtual Staging API")
    print("=" * 70)
    print("✓ AWS S3 Integration enabled")
    print("✓ Version History & Save/Revert enabled")
    print("✓ Image Generation with Gemini enabled")
    print("✓ CORS enabled for cross-origin requests")
    print("=" * 70)
    print("\n📍 Server running on: http://localhost:5000")
    print("\n📚 Virtual Staging Endpoints:")
    print("   • POST   /api/virtual-staging/session - Create session")
    print("   • GET    /api/virtual-staging/session/<id> - Get session")
    print("   • DEL    /api/virtual-staging/session/<id> - Delete session")
    print("   • POST   /api/virtual-staging/generate - Generate image")
    print("   • POST   /api/virtual-staging/refine - Refine image")
    print("   • POST   /api/virtual-staging/save-change - Save version")
    print("   • POST   /api/virtual-staging/revert-change - Revert version")
    print("   • GET    /api/virtual-staging/version-history/<id> - Get history")
    print("   • GET    /api/virtual-staging/property/<id> - Get property sessions")
    print("   • GET    /api/virtual-staging/user/<id> - Get user sessions")
    print("   • GET    /api/virtual-staging/styles - Get available styles")
    print("   • GET    /api/virtual-staging/furniture-themes - Get furniture themes")
    print("   • GET    /api/virtual-staging/color-palettes - Get color palettes")
    print("\n📦 AWS S3 Management Endpoints:")
    print("   • POST   /api/aws-s3/upload - Upload file to S3")
    print("   • GET    /api/aws-s3/list/<folder> - List files in folder")
    print("   • GET    /api/aws-s3/download/<key> - Download file")
    print("   • DEL    /api/aws-s3/delete/<key> - Delete file")
    print("   • GET    /api/aws-s3/exists/<key> - Check if file exists")
    print("   • GET    /api/aws-s3/url/<key> - Get file URL")
    print("   • GET    /api/aws-s3/info/<key> - Get file metadata")
    print("   • POST   /api/aws-s3/batch/delete - Delete multiple files")
    print("   • POST   /api/aws-s3/batch/info - Get info for multiple files")
    print("\n🏥 Health Check:")
    print("   • GET    /health - Health check")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
