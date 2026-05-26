# Deployment Instructions

To deploy the system, follow the steps from **iContinuum Example 3**:  
https://github.com/disnetlab/iContinuum/tree/master/Example3

## Adjustments Required

### 1. Deployment Images
The updated container images referenced in `deployment.yml.j2` will be added after submission.  
Change the db image to cvetac/dbmicroservice:latest2
Change MS1 image to lacki/microservice1:v1

### 2. Inventory Configuration
Edit `inventory.invi` and update the placeholder IP addresses with your own IPs.