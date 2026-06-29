from distutils.core import setup

setup(name='dlab',
      version='0.0',
      description='denmanlab code, mostly for neurophysiology',
      author='denman lab',
      author_email='daniel.denman@cuanschutz.edu',
      url='https://denmanlab.github.io',
      packages= ['dlab'],
      install_requires = ['scipy', 'matplotlib', 'numpy','pandas','scikit-learn'],
     )