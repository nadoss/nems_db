##  Flask settings  ##

# Enable debug mode - should *NOT* be used for a publicly-accessible site
DEBUG = "FALSE"
# Enable request forgery protection
CSRF_ENABLED = "TRUE"
# Secure key for signing CSRF data
CSRF_SESSION_KEY = "CHANGE ME TO SOMETHING SUPER SECRET"
# Secure key for signing cookies
SECRET_KEY = "ALSO CHANGE ME TO SOMETHING JUST AS SECRET"
# PORT for web server
PORT = "8000"
