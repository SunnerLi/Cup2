from lxml import etree
import urllib2
import os

xml_dir = './xml/'
dat_dir = './dat/'

def parse(file_name):
    """
        Parse the PASCAL-format description file and transfer into my own format

        Arg:    The name of PASCAL-format description file
        Ret:    The information string
    """
    info = etree.iterparse(file_name)
    string = ""
    for tag, element in info:
        if element.tag == "name" or \
            element.tag == 'xmin' or \
            element.tag == 'ymin' or \
            element.tag == 'xmax' or \
            element.tag == 'ymax':
            string = string + element.text + '\n'
    info = str(len(string.split('\n'))/5) + '\n'
    info += string[:-1]
    return info

if __name__ == '__main__':
    for name in os.listdir(xml_dir):
        if name[-4:] == '.xml':
            with open(dat_dir + name[:-4] + '.dat', 'w') as f:
                f.write(parse(xml_dir + name))