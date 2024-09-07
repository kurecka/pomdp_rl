ARG paynt_base=randriu/paynt
# ARG paynt_base=xkurecka/paynt:
FROM $paynt_base

RUN pip install tqdm dill matplotlib pandas seaborn
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install skrl pytest gymnasium scipy
WORKDIR /opt/learning
