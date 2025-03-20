# Import necessary modules from Flask
from flask import request, Response

# Define a function to restrict access to a Flask app based on IP whitelisting
def ip_whitelist(app, whitelisted_ips):
    """
    Restrict access to a Flask app based on IP whitelisting.

    Args:
        app (Flask): The Flask app to restrict access to.
        whitelisted_ips (list): A list of IP addresses that are allowed to access the app.

    Returns:
        None
    """

    # Use the @app.before_request decorator to execute the restrict_ip function before each request
    @app.before_request
    def restrict_ip():
        """
        Check if the IP address of the incoming request is in the whitelist.

        Returns:
            Response: A 403 Forbidden response if the IP address is not in the whitelist.
        """

        # Get the IP address of the incoming request
        user_ip = request.remote_addr
        print(user_ip, "-------")  # Log the IP address for debugging purposes

        # Check if the IP address is in the whitelist
        if user_ip not in whitelisted_ips:
            print(user_ip, "****")  # Log the IP address if it's not in the whitelist
            # Return a 403 Forbidden response if the IP address is not in the whitelist
            return Response(f"Access denied for IP: {user_ip}", status=403)