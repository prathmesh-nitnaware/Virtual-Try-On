import sys
import zipfile
import xml.etree.ElementTree as ET

def get_docx_text(path):
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.XML(xml_content)
    
    WORDS_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = WORDS_NAMESPACE + 'p'
    TEXT = WORDS_NAMESPACE + 't'
    
    paragraphs = []
    for paragraph in tree.iter(PARA):
        texts = [node.text
                 for node in paragraph.iter(TEXT)
                 if node.text]
        if texts:
            paragraphs.append("".join(texts))
            
    return '\n'.join(paragraphs)

if __name__ == '__main__':
    text = get_docx_text(sys.argv[1])
    with open('plan.txt', 'w', encoding='utf-8') as f:
        f.write(text)
