from distutils.core import setup
setup(
  name = 'TFST',         # How you named your package folder (MyLib)
  packages = ['TFST'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Time-Frequency-Scale Transform',   # Give a short description about your library
  author = 'Gonzalo Romero-García',                   # Type in your name
  author_email = 'tritery@hotmail.com',      # Type in your E-Mail
  url = 'https://github.com/Manza12/TFST',   # Provide either the link to your github or to your website
  #download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Time', 'Frequency', 'Scale', 'Transform', 'Signal Processing'],   # Keywords that define your package best
  install_requires=[],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
  ],
)
