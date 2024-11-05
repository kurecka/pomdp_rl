This folder describes a Docker image that can be use at the E-Infra Jupyter Hub.
The docker has pre-installed Jupyter, Paynt, and VecStrom.

## How to use the precompiled image
The image is available at dockerhub under the tag `kurecka/paynt:jupyter`. To use it, follow these steps:
1. Log in to the E-Infra Jupyter Hub at [https://hub.cloud.e-infra.cz/hub/login](https://hub.cloud.e-infra.cz/hub/login)`.
2. Click `Home` in the top menu.
3. Name your server and click `Add New Server`.
4. Choosing image: Select `Custom image` and fill `kurecka/paynt:jupyter` into the name field.
   1. Note that you can also check the checkbox `Ensure ssh access into the notebook` to enable SSH access to the notebook.
5. Choosing storage: Select the storage you want to use.
   1. You can check the box `Mount MetaCentrum home` to get access to your MetaCentrum home directory. You will be asked to choose the frontend server. See [the docs](https://docs.metacentrum.cz/computing/frontends/) for the list of available homes and their respective frontend domains. The mounted home will be available at `/home/meta/username`.
6. Choose resources: Select the resources you need. __If you choose a GPU, do not forget to stop the server when you are done__!
7. Press `Start` to start the server.
