import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tempfile
import subprocess
import shutil
import gzip
from astropy.table import Table

import requests

import barbershop
import glob
import os
import sys

'''A script that downloads TGAS data for me,
totally stolen off Jo Bovys gaia_tools.'''

def get_TGAS(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'Gaia_TGAS') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'Gaia_TGAS')

    #Building local and remote paths
    local = [os.path.join(dirloc, 'Gaia_TGAS', 'TgasSource_000-000-%03i.csv.gz' % ii)
                for ii in range(16)]
    remote = [os.path.join('http://cdn.gea.esac.esa.int','Gaia','tgas_source','csv',
                'TgasSource_000-000-%03i.csv.gz' % ii) for ii in range(16)]

    #Download the data
    for localpath, remotepath in list(zip(local, remote)):
        #Check data doesnt already exist
        if not os.path.isfile(localpath):
            download(localpath, remotepath, verbose=True)
        else:
            sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
            sys.stdout.flush()

def get_KIC(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'KIC') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'KIC')

    #Building local and remote paths
    localpath = os.path.join(dirloc, 'KIC', 'kic.txt.gz')
    remotepath = os.path.join('http://archive.stsci.edu/pub/kepler/catalogs/','kic.txt.gz')

    #Check data doesnt already exist
    if not os.path.isfile(localpath):
        download(localpath, remotepath, verbose=True)
    else:
        sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
        sys.stdout.flush()

def get_2MASSxTGAS(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + '2MASSxTGAS') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'2MASSxTGAS')

    localpath = os.path.join(dirloc, '2MASSxTGAS', 'tgas-matched-2mass.fits.gz')
    remotepath = 'http://portal.nersc.gov/project/cosmo/temp/dstn/gaia/tgas-matched-2mass.fits.gz'

    if not os.path.isfile(localpath):
        download(localpath, remotepath, verbose=True)
    else:
        sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
        sys.stdout.flush()

def get_DR2(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'Gaia_DR2') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'Gaia_DR2')

    #Building local and remote paths
    local = [os.path.join(dirloc, 'Gaia_DR2', 'DR2_000-000-%03i.csv' % ii)
                for ii in range(16)]
    remote = [os.path.join('http://cdn.gea.esac.esa.int','Gaia','DR2','csv',
                'DR2_000-000-%03i.csv.gz' % ii) for ii in range(16)]

    #Download the data
    for localpath, remotepath in list(zip(local, remote)):
        if not os.path.isfile(localpath):
            download(localpath, remotepath, verbose=True)
        else:
            sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
            sys.stdout.flush()

def download(localpath, remotepath, verbose=False):
    sys.stdout.write('\r'+"Downloading file %s ...\r" % (os.path.basename(remotepath)))
    sys.stdout.flush()

    downloading = True
    interrupted = False
    #Create a temporary file
    file, tmp_savefilename = tempfile.mkstemp()
    os.close(file)
    ntries = 1

    while downloading:
        try:
            cmd = ['wget','%s' % remotepath,
                    '-O', '%s' % tmp_savefilename,
                    '--read-timeout=10',
                    '--tries=3']
            if not verbose:
                cmd.append('-q')

            subprocess.check_call(cmd) #Run command
            shutil.move(tmp_savefilename, localpath) #Move temp file to local
            downloading=False   #End download
            if interrupted:
                raise KeyboardInterrupt

        #Check for keyboardinterrupted or repeated attempts at failed download
        except subprocess.CalledProcessError as e:
            if not downloading: #Assume Keyboardinterrupt
                raise
            elif ntries > 2:
                raise IOError('File %s does not appear to exist on the server ...' % (os.path.basename(remotepath)))
            elif not 'exit status 4' in str(e):
                interrupted = True
            os.remove(tmp_savefilename)

        #Check for OSError due to missing wget
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                raise OSError("Automagically downloading catalogs requires the wget program; please install wget and try again...")
            else:
                raise OSError("Not quite sure whats gone wrong here...")

        #Just make sure the temp file is removed
        finally:
            if os.path.exists(tmp_savefilename):
                os.remove(tmp_savefilename)

        #Up the ntries
        ntries += 1
    sys.stdout.write('\r'+"Download Complete"+'\r')
    sys.stdout.flush()

    return None

def unzip(sfile, sep = ',', columns=None, check_info=False):
    '''A simple function that returns a pandas dataframe from a gzipped .csv
    file, and optionally returns chose columns '''
    file = gzip.open(sfile, 'rb')

    if '.csv' in os.path.basename(sfile):
        if check_info:
            print(pd.read_csv(file, sep=sep).info())

        if columns == None:
            return pd.read_csv(file, sep=sep)
        else:
            return pd.read_csv(file, sep=sep, usecols=columns)

    elif '.fits' in os.path.basename(sfile):
        dat = Table.read(sfile, format='fits')
        if check_info:
            print(dat.info())
        return dat.to_pandas()[columns]

    elif '.txt' in os.path.basename(sfile):
        if check_info:
            print(pd.read_csv(file, sep=sep).info())

        if columns == None:
            return pd.read_csv(file, sep=sep)
        else:
            return pd.read_csv(file, sep=sep, usecols=columns)

    else:
        print('Filetype not recognised (currently only .fits, .csv and .txt supported)')

def query_simbad():
    object_name = '16 Cyg A'

    results_table = Simbad.query_objectsids(object_name)

    verbose=True
    SIMBAD_URL = 'http://' + 'simbad.u-strasbg.fr' +'/simbad/sim_script'
    TIMEOUT = 60

    verify = True
    auth = True
    stream = False

    '''From ..query import BaseQuery, which has the _request function'''

    request_payload = dict(script="\n".join(('format object "%IDLIST"',
                                            'query id %s' % object_name)))

    session = requests.Session()


    '''
    method = 'POST'
    url = SIMBAD_URL
    data = request_payload
    timeout = TIMEOUT

    Lets now save for now, lets just call the data into python.
    '''
    response = session.request('POST', SIMBAD_URL, data=request_payload, timeout=TIMEOUT)

    requests.session().session.request(method, url, data=data, timeout=timeout, auth=auth, verify=verify)


    # query = AstroQuery(method, url, params=None, data=data, headers=None, files=None, timeout=timeout)
    # response = query.request(self._session, stream=stream, auth=auth, verify=verify)

class AstroQuery(object):
    def __init__(self, method, url, data, timeout):
        self.method = method
        self.url = url
        self.data = data
        self.timeout=timeout



if __name__ == "__main__":




    dirloc = '/home/oliver/PhD/Catalogues/'
    get_TGAS(dirloc)
    get_KIC(dirloc)

    #http://archive.stsci.edu/kepler/kic10/help/quickcol.html
    kk = os.path.join(dirloc, 'KIC', 'kic.txt.gz')
    columns = ['kic_kepler_id']
    kic = unzip(kk, sep='|', columns=columns)

    ll = os.path.join(dirloc, 'Gaia_TGAS','TgasSource_000-000-001.csv.gz')
    columns = ['hip','tycho2_id','source_id','parallax','parallax_error','phot_g_mean_mag']
    df = unzip(ll, sep=',', columns=columns)




    sys.exit()
    get_2MASSxTGAS(dirloc)
    tt = os.path.join(dirloc, '2MASSxTGAS','tgas-matched-2mass.fits.gz')
    columns = ['ra', 'dec']
    twomass = unzip(tt, columns=columns, check_info=True)
